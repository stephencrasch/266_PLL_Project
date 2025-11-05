import json
import tarfile
import os
from pathlib import Path
from typing import Iterator, Dict, Any
import pandas as pd


class YelpDataLoader:
    """Load and process Yelp dataset from tar file."""
    
    def __init__(self, tar_path: str, extract_dir: str = "yelp_data"):
        """
        Initialize the Yelp data loader.
        
        Args:
            tar_path: Path to the yelp_dataset.tar file
            extract_dir: Directory to extract files to (default: yelp_data)
        """
        self.tar_path = tar_path
        self.extract_dir = extract_dir
        self.extracted_files = {}
        
    def extract_tar(self):
        """Extract the tar file to the specified directory."""
        print(f"Extracting {self.tar_path}...")
        os.makedirs(self.extract_dir, exist_ok=True)
        
        with tarfile.open(self.tar_path, 'r') as tar:
            # Get list of files in tar
            members = tar.getmembers()
            print(f"Found {len(members)} files in archive")
            
            # Extract all files
            tar.extractall(path=self.extract_dir)
            
            # Store paths to extracted JSON files
            for member in members:
                if member.name.endswith('.json'):
                    file_path = os.path.join(self.extract_dir, member.name)
                    if 'business' in member.name.lower():
                        self.extracted_files['business'] = file_path
                    elif 'review' in member.name.lower():
                        self.extracted_files['review'] = file_path
                    elif 'user' in member.name.lower():
                        self.extracted_files['user'] = file_path
                    elif 'tip' in member.name.lower():
                        self.extracted_files['tip'] = file_path
                    elif 'checkin' in member.name.lower():
                        self.extracted_files['checkin'] = file_path
                        
        print(f"Extraction complete. Files available: {list(self.extracted_files.keys())}")
        return self.extracted_files
    
    def read_json_lines(self, file_path: str, max_records: int = None) -> Iterator[Dict[Any, Any]]:
        """
        Read JSON file line by line (Yelp format: one JSON object per line).
        
        Args:
            file_path: Path to JSON file
            max_records: Maximum number of records to read (None for all)
            
        Yields:
            Dictionary for each JSON object
        """
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_records and count >= max_records:
                    break
                try:
                    yield json.loads(line.strip())
                    count += 1
                    if count % 100000 == 0:
                        print(f"Processed {count} records...")
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    
    def load_businesses(self, max_records: int = None) -> pd.DataFrame:
        """
        Load business data into a pandas DataFrame.
        
        Args:
            max_records: Maximum number of records to load (None for all)
            
        Returns:
            DataFrame containing business data
        """
        if 'business' not in self.extracted_files:
            raise ValueError("Business file not found. Run extract_tar() first.")
        
        print(f"Loading businesses from {self.extracted_files['business']}...")
        businesses = list(self.read_json_lines(self.extracted_files['business'], max_records))
        df = pd.DataFrame(businesses)
        print(f"Loaded {len(df)} businesses")
        return df
    
    def load_reviews(self, max_records: int = None) -> pd.DataFrame:
        """
        Load review data into a pandas DataFrame.
        
        Args:
            max_records: Maximum number of records to load (None for all)
            
        Returns:
            DataFrame containing review data
        """
        if 'review' not in self.extracted_files:
            raise ValueError("Review file not found. Run extract_tar() first.")
        
        print(f"Loading reviews from {self.extracted_files['review']}...")
        reviews = list(self.read_json_lines(self.extracted_files['review'], max_records))
        df = pd.DataFrame(reviews)
        print(f"Loaded {len(df)} reviews")
        return df
    
    def load_businesses_chunked(self, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Load businesses in chunks for memory efficiency.
        
        Args:
            chunk_size: Number of records per chunk
            
        Yields:
            DataFrame chunks
        """
        if 'business' not in self.extracted_files:
            raise ValueError("Business file not found. Run extract_tar() first.")
        
        print(f"Loading businesses in chunks of {chunk_size}...")
        chunk = []
        for record in self.read_json_lines(self.extracted_files['business']):
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
        
        if chunk:  # Yield remaining records
            yield pd.DataFrame(chunk)
    
    def load_reviews_chunked(self, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Load reviews in chunks for memory efficiency.
        
        Args:
            chunk_size: Number of records per chunk
            
        Yields:
            DataFrame chunks
        """
        if 'review' not in self.extracted_files:
            raise ValueError("Review file not found. Run extract_tar() first.")
        
        print(f"Loading reviews in chunks of {chunk_size}...")
        chunk = []
        for record in self.read_json_lines(self.extracted_files['review']):
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
        
        if chunk:  # Yield remaining records
            yield pd.DataFrame(chunk)


def main():
    """Example usage of the YelpDataLoader."""
    
    # Initialize loader
    loader = YelpDataLoader(
        tar_path="yelp_dataset.tar",
        extract_dir="yelp_data"
    )
    
    # Extract the tar file (only needs to be done once)
    loader.extract_tar()
    
    # Option 1: Load all data into memory (only for smaller datasets or samples)
    print("\n--- Loading sample of businesses ---")
    businesses_df = loader.load_businesses(max_records=1000)
    print(f"\nBusiness columns: {businesses_df.columns.tolist()}")
    print(f"\nFirst business:\n{businesses_df.iloc[0]}")
    
    print("\n--- Loading sample of reviews ---")
    reviews_df = loader.load_reviews(max_records=1000)
    print(f"\nReview columns: {reviews_df.columns.tolist()}")
    print(f"\nFirst review:\n{reviews_df.iloc[0]}")
    
    # Option 2: Process data in chunks (recommended for large datasets)
    print("\n--- Processing businesses in chunks ---")
    business_count = 0
    for chunk in loader.load_businesses_chunked(chunk_size=10000):
        business_count += len(chunk)
        # Process chunk here (e.g., filter, transform, save to database)
        print(f"Processed {business_count} businesses so far...")
        if business_count >= 50000:  # Stop after 50k for demo
            break
    
    print("\n--- Processing reviews in chunks ---")
    review_count = 0
    for chunk in loader.load_reviews_chunked(chunk_size=10000):
        review_count += len(chunk)
        # Process chunk here (e.g., filter, transform, save to database)
        print(f"Processed {review_count} reviews so far...")
        if review_count >= 50000:  # Stop after 50k for demo
            break
    
    # Save samples to CSV for inspection
    print("\n--- Saving samples to CSV ---")
    businesses_df.to_csv("businesses_sample.csv", index=False)
    reviews_df.to_csv("reviews_sample.csv", index=False)
    print("Saved businesses_sample.csv and reviews_sample.csv")


if __name__ == "__main__":
    main()
