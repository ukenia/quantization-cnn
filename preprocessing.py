import os
import sentencepiece as spm


# transcripts are in the binary bit format so we would need to have string format before sentencepiece.
BASE_PATH = "/content/gdrive/MyDrive/idl_proj/"
DATA_PATH = os.path.join(BASE_PATH, "data")

# for training dataset
train_transcripts_filename = os.path.join(DATA_PATH, 'train_transcripts.npy')
dev_transcripts_filename = os.path.join(DATA_PATH, 'dev_transcripts.npy')

train_transcripts_cleaned_filename = os.path.join(DATA_PATH, 'train_transcripts_cleaned.txt')
dev_transcripts_cleaned_filename = os.path.join(DATA_PATH, 'dev_transcripts_cleaned.txt')


train_transcripts = np.load(train_transcripts_filename, allow_pickle=True)
train_transcrips_decoded = []
for i, t in enumerate(train_transcripts):
  s = [i.decode() for i in t]
  s = (' '.join(s)).strip()
  train_transcrips_decoded.append(s)
print("Done", len(train_transcrips_decoded))
with open(train_transcripts_cleaned_filename, 'w') as file:
  file.write('\n'.join(train_transcrips_decoded))


dev_transcripts = np.load(dev_transcripts_filename, allow_pickle=True)
dev_transcrips_decoded = []
for i, t in enumerate(dev_transcripts):
  s = [i.decode() for i in t]
  s = (' '.join(s)).strip()
  dev_transcrips_decoded.append(s)
print("Done", len(dev_transcrips_decoded))
with open(dev_transcripts_cleaned_filename, 'w') as file:
  file.write('\n'.join(dev_transcrips_decoded))


# train a global deterministic character level transcripts tokenizer on the cleaned text files
spm.SentencePieceTrainer.train(f'--input={train_transcripts_cleaned_filename} --model_prefix=train_model_cleaned --vocab_size=200 --character_coverage=1.0 --model_type=char')

print("Trained an SPM model")
