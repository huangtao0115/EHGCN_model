import openai
import pandas as pd
import numpy as np
import time
import os
import csv
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from openai.error import APIError, RateLimitError, APIConnectionError, AuthenticationError, Timeout

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_API_KEY = "you key"


openai.api_base = OPENAI_API_BASE
openai.api_key = OPENAI_API_KEY

BATCH_SIZE = 500
MAX_RETRIES = 5
INITIAL_DELAY = 1
BACKOFF_FACTOR = 2
RESULT_CSV = "gpt35_test_predictions.csv"
LOG_FILE = "processing_log.txt"
PROGRESS_FILE = "progress_checkpoint.txt"
MODEL_NAME = "gpt-3.5-turbo"  # model_name     gpt-4o-mini    gpt-3.5-turbo


train_df = pd.read_excel("train_new.xlsx", header=None, names=["text", "label"])
test_df = pd.read_excel("test_new.xlsx", header=None, names=["text", "label"])

test_df = test_df.reset_index(drop=False).rename(columns={'index': 'orig_index'})

def create_few_shot_messages(train_df, num_samples_per_label=10):

    unique_labels = train_df['label'].unique()

    system_prompt = (
        "Please classify the following text and return only the category label："
        + ", ".join([str(label) for label in unique_labels]) + ".\n"
    )

    few_shot_messages = [{"role": "system", "content": system_prompt}]

    examples = train_df.groupby("label").apply(
        lambda x: x.sample(n=num_samples_per_label, random_state=42)).reset_index(drop=True)

    for _, row in examples.iterrows():
        few_shot_messages.append({"role": "user", "content": row['text']})
        few_shot_messages.append({"role": "assistant", "content": str(row['label'])})  # 确保标签是字符串

    return few_shot_messages


few_shot_messages = create_few_shot_messages(train_df, num_samples_per_label=10)

def classify_with_retry(text, few_shot_messages):
    messages = few_shot_messages + [{"role": "user", "content": text}]

    for attempt in range(MAX_RETRIES):
        try:

            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
                request_timeout=30
            )


            reply = response['choices'][0]['message']['content'].strip()
            return reply

        except (APIError, RateLimitError, APIConnectionError, AuthenticationError, Timeout,
                requests.exceptions.SSLError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:

            delay = INITIAL_DELAY * (BACKOFF_FACTOR ** attempt)
            print(f"\n⚠️ Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)[:150]}...")
            print(f"🕒 Retrying in {delay} seconds...")
            time.sleep(delay)


        except Exception as e:
            error_msg = str(e)

            if "content policy" in error_msg.lower() or "violation" in error_msg.lower():
                print(f"\n🚫 Content policy violation for record: {error_msg[:150]}")
                return "violation_error"
            else:
                print(f"\n❌ Unrecoverable error: {str(e)[:150]}")
                return "error"

    print(f"🚨 Request failed after {MAX_RETRIES} attempts")
    return "error"

def init_csv_file():
    if not os.path.exists(RESULT_CSV):
        try:
            with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['orig_index', 'text', 'label', 'pred', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"✅ Created new result file: {RESULT_CSV}")
        except PermissionError as e:
            print(f"❌ Permission denied when creating file: {e}")
            print("⚠️ Please close any programs that may be using this file and restart the script")
            exit(1)
    else:
        print(f"✅ Using existing result file: {RESULT_CSV}")

def save_single_result(orig_index, text, label, pred):
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['orig_index', 'text', 'label', 'pred', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'orig_index': orig_index,
                    'text': text,
                    'label': label,
                    'pred': pred,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
            return

        except PermissionError as e:
            if attempt < max_retries - 1:
                print(
                    f"⚠️ Permission denied when saving record {orig_index}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"❌ Failed to save record {orig_index} after {max_retries} attempts: {e}")
                with open("save_errors.log", "a") as error_log:
                    error_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to save record {orig_index}: {e}\n")

        except Exception as e:
            print(f"❌ Unexpected error when saving record {orig_index}: {e}")
            with open("save_errors.log", "a") as error_log:
                error_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error saving record {orig_index}: {e}\n")
            break

def get_processed_indices():
    processed_indices = set()
    if os.path.exists(RESULT_CSV):
        try:
            with open(RESULT_CSV, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['orig_index'] and row['orig_index'].isdigit():
                        processed_indices.add(int(row['orig_index']))
        except Exception as e:
            print(f"⚠️ Error reading result file: {e}")
    return processed_indices

def save_progress_checkpoint(last_index):
    try:
        with open(PROGRESS_FILE, 'w') as f:
            f.write(str(last_index))
    except Exception as e:
        print(f"⚠️ Error saving progress checkpoint: {e}")

def load_progress_checkpoint():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

init_csv_file()

processed_indices = get_processed_indices()
print(f"ℹ️ Found {len(processed_indices)} processed records")

start_index = load_progress_checkpoint()
print(f"ℹ️ Starting from index: {start_index}")

total_samples = len(test_df)
num_batches = int(np.ceil(total_samples / BATCH_SIZE))

print(f"\n🚀 Starting processing of {total_samples} records")
print(f"📦 Batch size: {BATCH_SIZE}, Total batches: {num_batches}")
print(f"🧠 Using model: {MODEL_NAME}")

pbar_total = tqdm(total=total_samples, desc="Overall Progress", position=0)

pbar_total.update(len(processed_indices))

for batch_idx in range(int(np.ceil(start_index / BATCH_SIZE)), num_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min((batch_idx + 1) * BATCH_SIZE, total_samples)

    pbar_batch = tqdm(total=batch_end - batch_start, desc=f"Batch {batch_idx + 1}/{num_batches}", position=1)

    print(f"\n🔧 Processing batch {batch_idx + 1}/{num_batches} (records {batch_start}-{batch_end - 1})")

    for idx in range(batch_start, batch_end):
        if idx in processed_indices:
            pbar_batch.update(1)
            pbar_total.update(1)
            continue

        row = test_df.iloc[idx]
        orig_index = row['orig_index']
        text = row['text']
        label = row['label']

        pred = classify_with_retry(text, few_shot_messages)

        if pred in ["error", "violation_error"]:
            print(f"⚠️ Error detected for record {orig_index}. Saving 'error'.")
            pred = "error"

        save_single_result(orig_index, text, label, pred)

        pbar_batch.update(1)
        pbar_total.update(1)

        if idx % 10 == 0:
            save_progress_checkpoint(idx)

    pbar_batch.close()

    save_progress_checkpoint(batch_end)

    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"Batch {batch_idx + 1}: Processed {batch_start}-{batch_end - 1} at {time.ctime()}\n")
    except Exception as e:
        print(f"⚠️ Error writing to log file: {e}")

    print(f"💾 Saved results for batch {batch_idx + 1}")

pbar_total.close()

if os.path.exists(PROGRESS_FILE):
    try:
        os.remove(PROGRESS_FILE)
        print("✅ Removed progress checkpoint file")
    except Exception as e:
        print(f"⚠️ Error removing progress file: {e}")

print("\n📊 All data processed, starting evaluation...")

try:
    result_df = pd.read_csv(RESULT_CSV)
except FileNotFoundError:
    print("❌ Result file not found. Evaluation skipped.")
    exit(1)

valid_df = result_df[result_df['pred'] != 'error']
error_count = (result_df['pred'] == 'error').sum()

if len(valid_df) > 0:
    y_true = valid_df["label"].astype(str).str.strip().str.lower()
    y_pred = valid_df["pred"].astype(str).str.strip().str.lower()

    unique_labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\n🎯 Final Evaluation Results:")
    print(f"✅ Valid samples: {len(valid_df)}/{len(result_df)}")
    print(f"❌ Error/failed samples: {error_count}")
    print("🎯 Accuracy:", acc)
    print("🎯 Macro F1 Score:", f1)
    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_labels))

    try:
        result_file = f"{MODEL_NAME.replace('-', '_')}_predictions_final.xlsx"
        result_df.to_excel(result_file, index=False)
        print(f"\n✅ Final results saved to {result_file}")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
else:
    print("❌ No valid results available for evaluation")

detailed_report = f"{MODEL_NAME.replace('-', '_')}_classification_report.txt"
try:
    with open(detailed_report, 'w') as f:
        f.write(f"Dataset size: {len(result_df)}\n")
        f.write(f"Valid samples: {len(valid_df)}\n")
        f.write(f"Error samples: {error_count}\n\n")
        if len(valid_df) > 0:
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, labels=unique_labels))
    print(f"📝 Detailed report saved to {detailed_report}")
except Exception as e:
    print(f"❌ Error saving detailed report: {e}")

print("\n✅ Processing complete!")