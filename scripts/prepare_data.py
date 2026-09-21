from datasets import load_dataset
from weird_ai.config import RAW_DATA_DIR, SAMPLE_LYRICS_FILE, PROCESSED_DATA_DIR

"""
Read the downloaded dataset (parquet files) and create lyrics_sample.txt.
"""
def main():
    #gets songs and puts a section with the limit into an array
    lyrics_column = "lyrics"
    limit = 5000
    selected_lyrics = []

    #makes a directory for the processed data if it doesn't already exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    #loads the dataset
    dataset = load_dataset(
        "parquet",
        data_files=str(RAW_DATA_DIR / "*.parquet"),
        split="train"
    )

    print(dataset)
    print(dataset.column_names)

    #iterates through the dataset, processed the data
    for row in dataset:
        lyrics = row.get(lyrics_column)

        if lyrics is None:
            continue

        lyrics = lyrics.strip()

        if len(lyrics) < 100:
            continue

        selected_lyrics.append(lyrics)

        if len(selected_lyrics) >= limit:
            break

    #structuring songs with formatted lyrics 
    output_text = "\n\n<|song|>\n\n".join(selected_lyrics)

    #writes the output to a file 
    SAMPLE_LYRICS_FILE.write_text(output_text, encoding="utf-8")

    print(f"Songs written: {len(selected_lyrics)}")
    print(f"Output file: {SAMPLE_LYRICS_FILE}")


if __name__ == "__main__":
    main()
