import pyopenjtalk

text = "ちょっと待ってって言ったのにどうして行っちゃうの"
print(f"Text: {text}")
print("\nFull G2P output:")
phones = pyopenjtalk.g2p(text, join=False)
print(f"Phonemes: {phones}")

# Check what pyopenjtalk outputs for small tsu
test_words = ["ちょっと", "待って", "言った", "行っちゃう"]
for word in test_words:
    p = pyopenjtalk.g2p(word, join=False)
    print(f"\n{word}: {p}")

