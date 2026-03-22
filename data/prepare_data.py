import pandas as pd

def prepare():
    input_path = "G:/Dsbda/tweets.csv"
    output_path = "G:/Dsbda/sentiment-analysis/data/tweets.csv"
    
    print(f"Loading data from {input_path}...")
    # The file has no headers, load it and assign columns
    df = pd.read_csv(input_path, header=None, names=['text', 'sentiment'])
    
    # Filter only positive and negative
    df = df[df['sentiment'].isin(['Positive', 'Negative'])]
    
    # Keep only target columns
    df = df[['text', 'sentiment']]
    
    # Drop NAs
    df = df.dropna()
    
    # Save the prepared data
    df.to_csv(output_path, index=False)
    print(f"Prepared data saved to {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    prepare()
