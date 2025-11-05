"""
Script to prepare Yelp data for embeddings generation.
This will extract and clean review text and business categories.
"""

import pandas as pd
from typing import List
from load_yelp_data import YelpDataLoader


class EmbeddingsDataPreparation:
    """Prepare Yelp data for embeddings generation."""
    
    # Target bar categories
    TARGET_CATEGORIES = {
        'Cocktail Bars',
        'Dive Bars', 
        'Pubs',
        'Sports Bars',
        'Wine Bars'
    }
    
    def __init__(self, loader: YelpDataLoader):
        self.loader = loader
        
    def parse_categories(self, categories_str: str) -> List[str]:
        if pd.isna(categories_str) or not categories_str:
            return []
        return [cat.strip() for cat in categories_str.split(',')]
    
    def clean_review_text(self, text: str) -> str:
        """
        Clean review text by normalizing all whitespace to single spaces.
        Converts text to a single line with normalized spacing.
        """
        if pd.isna(text):
            return ""
        
        import re
        # Replace all whitespace (newlines, tabs, multiple spaces) with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def prepare_review_embeddings_data(self, max_records: int = None, 
                                       min_text_length: int = 25,
                                       max_text_length: int = None,
                                       min_stars: float = None,
                                       max_stars: float = None,
                                       min_useful: int = None,
                                       sample_size: int = None,
                                       random_state: int = 42,
                                       business_ids: List[str] = None,
                                       businesses_df: pd.DataFrame = None,
                                       save_path: str = None) -> pd.DataFrame:
        """
        Prepare review data for embeddings.
        
        Args:
            business_ids: Optional list of business IDs to filter reviews by
            businesses_df: Optional businesses DataFrame with target_categories for labeling
        """
        reviews_df = self.loader.load_reviews(max_records=max_records)
        
        # Filter by business IDs first (if provided)
        if business_ids is not None:
            print(f"Filtering reviews to {len(business_ids)} businesses...")
            filtered_df = reviews_df[reviews_df['business_id'].isin(business_ids)].copy()
            print(f"Found {len(filtered_df)} reviews from filtered businesses")
        else:
            filtered_df = reviews_df.copy()
        
        # Clean review text
        print("Cleaning review text (normalizing whitespace)...")
        filtered_df['text'] = filtered_df['text'].apply(self.clean_review_text)
        
        filtered_df['text_length'] = filtered_df['text'].str.len()
        
        # Apply other filters
        filtered_df = filtered_df[filtered_df['text_length'] >= min_text_length]
        if max_text_length:
            filtered_df = filtered_df[filtered_df['text_length'] <= max_text_length]
        if min_stars is not None:
            filtered_df = filtered_df[filtered_df['stars'] >= min_stars]
        if max_stars is not None:
            filtered_df = filtered_df[filtered_df['stars'] <= max_stars]
        if min_useful is not None:
            filtered_df = filtered_df[filtered_df['useful'] >= min_useful]
        
        # Random sample if specified
        if sample_size is not None and sample_size < len(filtered_df):
            filtered_df = filtered_df.sample(n=sample_size, random_state=random_state)
        
        # Add labels AND business names from businesses_df if provided
        if businesses_df is not None and 'target_categories' in businesses_df.columns:
            print("Adding labels and business names from business data...")
            # Merge to get target categories AND business names - use INNER join to only keep reviews from businesses in our filtered list
            label_df = businesses_df[['business_id', 'target_categories', 'name']].copy()
            filtered_df = filtered_df.merge(label_df, on='business_id', how='inner')
            
            print(f"After merging with businesses: {len(filtered_df)} reviews")
            
            # Create label (first target category) and label_count
            filtered_df['label'] = filtered_df['target_categories'].apply(
                lambda cats: cats[0] if isinstance(cats, list) and len(cats) > 0 else None
            )
            filtered_df['label_count'] = filtered_df['target_categories'].apply(
                lambda cats: len(cats) if isinstance(cats, list) else 0
            )
            
            # Drop the target_categories column (we just needed it for creating label)
            filtered_df = filtered_df.drop(columns=['target_categories'])
            
            # Concatenate business name into review text
            print("Concatenating business names into review text...")
            filtered_df['text'] = filtered_df.apply(
                lambda row: f"{row['name']}. {row['text']}", 
                axis=1
            )
            # Update text length after adding business name
            filtered_df['text_length'] = filtered_df['text'].str.len()
            
            print(f"Labels assigned: {filtered_df['label'].value_counts().to_dict()}")
            
            # Verify no null labels
            null_labels = filtered_df['label'].isna().sum()
            if null_labels > 0:
                print(f"WARNING: {null_labels} reviews have null labels!")
            else:
                print(f"✓ All {len(filtered_df)} reviews have valid labels")
        
        # Select relevant columns (no need to include 'name' separately anymore, it's in the text)
        base_cols = ['review_id', 'business_id', 'user_id', 'stars', 'useful', 'funny', 'cool', 'text', 'text_length', 'date']
        if 'label' in filtered_df.columns:
            base_cols.extend(['label', 'label_count'])
        
        result_df = filtered_df[base_cols].copy()
        
        if save_path:
            result_df.to_csv(save_path, index=False)
        
        return result_df
    
    def prepare_business_embeddings_data(self, max_records: int = None,
                                        max_categories: int = None,
                                        save_path: str = None,
                                        filter_by_target_categories: bool = True) -> pd.DataFrame:
        businesses_df = self.loader.load_businesses(max_records=max_records)
        
        # Parse categories
        businesses_df['categories_list'] = businesses_df['categories'].apply(self.parse_categories)
        businesses_df['category_count'] = businesses_df['categories_list'].apply(len)
        
        # Filter out businesses without categories
        filtered_df = businesses_df[businesses_df['category_count'] > 0].copy()
        
        # Filter to only TARGET_CATEGORIES if specified
        if filter_by_target_categories:
            print(f"Filtering businesses to only include target categories: {self.TARGET_CATEGORIES}")
            # Keep only businesses that have at least one target category
            filtered_df['has_target_category'] = filtered_df['categories_list'].apply(
                lambda cats: any(cat in self.TARGET_CATEGORIES for cat in cats)
            )
            filtered_df = filtered_df[filtered_df['has_target_category']].copy()
            filtered_df = filtered_df.drop(columns=['has_target_category'])
            print(f"Found {len(filtered_df)} businesses with target categories")
            
            # Add target categories only (for labeling)
            filtered_df['target_categories'] = filtered_df['categories_list'].apply(
                lambda cats: [cat for cat in cats if cat in self.TARGET_CATEGORIES]
            )
            filtered_df['target_category_count'] = filtered_df['target_categories'].apply(len)
            
            # Filter to only businesses with exactly ONE target category (no ambiguity)
            print("Filtering to businesses with exactly 1 target category (removing ambiguous multi-label businesses)...")
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['target_category_count'] == 1].copy()
            removed_count = before_count - len(filtered_df)
            print(f"Kept {len(filtered_df)} businesses with single target category (removed {removed_count} ambiguous businesses)")
        
        # Filter by maximum number of categories if specified
        if max_categories is not None:
            filtered_df = filtered_df[filtered_df['category_count'] <= max_categories]
        
        # Select relevant columns
        base_cols = ['business_id', 'name', 'city', 'state', 'stars', 
                     'review_count', 'categories', 'categories_list', 
                     'category_count']
        if filter_by_target_categories:
            base_cols.extend(['target_categories', 'target_category_count'])
        
        result_df = filtered_df[base_cols].copy()
        
        if save_path:
            result_df.to_csv(save_path, index=False)
        
        return result_df
    
    def create_category_embeddings_data(self, businesses_df: pd.DataFrame = None,
                                       save_path: str = None) -> pd.DataFrame:
        
        if businesses_df is None:
            businesses_df = self.loader.load_businesses()
            businesses_df['categories_list'] = businesses_df['categories'].apply(self.parse_categories)
        
        # Explode categories so each category gets its own row
        exploded = businesses_df.explode('categories_list')
        exploded = exploded[exploded['categories_list'].notna()].copy()
        
        # Aggregate by category
        category_stats = exploded.groupby('categories_list').agg({
            'business_id': 'count',
            'stars': 'mean',
            'review_count': 'sum'
        }).reset_index()
        
        category_stats.columns = ['category', 'business_count', 'avg_stars', 'total_reviews']
        category_stats = category_stats.sort_values('business_count', ascending=False)
        
        if save_path:
            category_stats.to_csv(save_path, index=False)
        
        return category_stats
    
    def prepare_joined_data(self, max_businesses: int = None, 
                           max_reviews_per_business: int = None,
                           save_path: str = None) -> pd.DataFrame:
        # Load data
        businesses_df = self.loader.load_businesses(max_records=max_businesses)
        businesses_df['categories_list'] = businesses_df['categories'].apply(self.parse_categories)
        
        reviews_df = self.loader.load_reviews()
        
        # Select relevant business columns
        business_cols = businesses_df[['business_id', 'name', 'city', 'state', 
                                       'stars', 'categories', 'categories_list']]
        business_cols = business_cols.rename(columns={'stars': 'business_stars'})
        
        # Join
        joined_df = reviews_df.merge(business_cols, on='business_id', how='inner')
        
        # Limit reviews per business if specified
        if max_reviews_per_business:
            joined_df = joined_df.groupby('business_id').head(max_reviews_per_business)
        
        # Add text length
        joined_df['text_length'] = joined_df['text'].str.len()
        
        if save_path:
            joined_df.to_csv(save_path, index=False)
        
        return joined_df


def main():
    """Example usage for preparing embeddings data."""
    
    # Initialize loader
    loader = YelpDataLoader(
        tar_path="yelp_dataset.tar",
        extract_dir="yelp_data"
    )
    
    # Extract if not already done
    if not loader.extracted_files:
        loader.extract_tar()
    else:
        # If already extracted, populate the extracted_files dict
        from pathlib import Path
        data_dir = Path(loader.extract_dir)
        if data_dir.exists():
            for file in data_dir.glob("*.json"):
                if 'business' in file.name.lower():
                    loader.extracted_files['business'] = str(file)
                elif 'review' in file.name.lower():
                    loader.extracted_files['review'] = str(file)
    
    # Initialize embeddings preparation
    prep = EmbeddingsDataPreparation(loader)
    
    # Prepare businesses with filters
    businesses_for_embeddings = prep.prepare_business_embeddings_data(
        max_records=None,
        max_categories=None,
        filter_by_target_categories=True,  # Only keep target bar categories
        save_path="businesses_for_embeddings.csv"
    )
    
    print(f"\nPrepared {len(businesses_for_embeddings)} businesses")
    print(f"Target category distribution:")
    print(businesses_for_embeddings['target_category_count'].value_counts().sort_index())
    
    # Prepare reviews ONLY from filtered businesses
    business_ids_to_include = businesses_for_embeddings['business_id'].tolist()
    print(f"\nFiltering reviews to only include {len(business_ids_to_include)} businesses...")
    
    reviews_for_embeddings = prep.prepare_review_embeddings_data(
        max_records=None,
        min_text_length=25,
        max_text_length=500,
        business_ids=business_ids_to_include,  # Only reviews from filtered businesses
        businesses_df=businesses_for_embeddings,  # Pass businesses for labeling
        sample_size=None,
        save_path="reviews_for_embeddings.csv"
    )
    
    print(f"\nPrepared {len(reviews_for_embeddings)} reviews")
    if 'label' in reviews_for_embeddings.columns:
        print(f"\nLabel distribution:")
        print(reviews_for_embeddings['label'].value_counts())
        print(f"\nLabel count distribution (how many categories per business):")
        print(reviews_for_embeddings['label_count'].value_counts().sort_index())
    
    # Create category statistics
    prep.create_category_embeddings_data(
        businesses_df=businesses_for_embeddings,
        save_path="categories_for_embeddings.csv"
    )
    
if __name__ == "__main__":
    main()
