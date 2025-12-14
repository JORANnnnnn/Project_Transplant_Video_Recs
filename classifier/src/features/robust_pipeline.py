"""
Robust Feature Extraction Pipeline for YouTube Transcripts
Processes data in batches with checkpoint saving and error recovery
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from integrated_features import create_extractor





class FeatureExtractionPipeline:
    """
    Robust pipeline for extracting features from YouTube transcripts with:
    - Batch processing
    - Checkpoint saving
    - Error recovery
    - Progress tracking
    """
    
    def __init__(self, 
                 db_path: str,
                 output_dir: str = "./feature_extraction_output",
                 batch_size: int = 50,
                 checkpoint_freq: int = 5,
                 load_medical_ner: bool = True,
                 load_spacy: bool = True):
        """
        Initialize pipeline.
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory for output files and checkpoints
            batch_size: Number of videos per batch
            checkpoint_freq: Save checkpoint every N batches
            load_medical_ner: Whether to load medical NER model
            load_spacy: Whether to load spaCy model
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_freq
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize feature extractor
        print("=" * 70)
        print("🚀 INITIALIZING FEATURE EXTRACTION PIPELINE")
        print("=" * 70)
        self.extractor = create_extractor(
            load_medical_ner=load_medical_ner,
            load_spacy=load_spacy
        )
        
        # State tracking
        self.processed_ids = set()
        self.failed_ids = {}
        self.results = []
        self.entity_records = []
        self.checkpoint_path = self.output_dir / "checkpoint.pkl"
        self.progress_path = self.output_dir / "progress.json"
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load previous checkpoint if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.processed_ids = checkpoint['processed_ids']
                    self.failed_ids = checkpoint['failed_ids']
                    self.results = checkpoint['results']
                    self.entity_records = checkpoint.get('entity_records', [])
                print(f"✅ Loaded checkpoint: {len(self.processed_ids)} already processed")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("   Starting fresh...")
    
    def _save_checkpoint(self, batch_num: int):
        """Save checkpoint."""
        checkpoint = {
            'processed_ids': self.processed_ids,
            'failed_ids': self.failed_ids,
            'results': self.results,
            'entity_records': self.entity_records,
            'timestamp': datetime.now().isoformat(),
            'batch_num': batch_num
        }
        
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"💾 Checkpoint saved (batch {batch_num})")
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
    
    def _save_progress(self, total: int, processed: int, failed: int):
        """Save progress statistics."""
        progress = {
            'total': total,
            'processed': processed,
            'failed': failed,
            'success_rate': (processed - failed) / processed * 100 if processed > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_path, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save progress: {e}")
    
    def _save_entity_counts(self, batch_num: int, entity_records: List[Dict]):
        """Save per-entity count records for a batch."""
        if not entity_records:
            return
        
        df = pd.DataFrame(entity_records).fillna(0)
        file = self.output_dir / f"batch_{batch_num:04d}_entity_counts.csv"
        try:
            df.to_csv(file, index=False)
            print(f"🧾 Entity counts saved to {file}")
        except Exception as e:
            print(f"⚠️  Failed to save entity counts for batch {batch_num}: {e}")
    
    def _build_entity_record(self, video_id: str, counts: Dict[str, int]) -> Dict[str, int]:
        """Create a flat record of entity counts for CSV export."""
        record = {'video_id': video_id, 'entity_total': sum(counts.values())}
        for label, value in counts.items():
            record[f"count_{label.lower()}"] = value
        return record
    
    def _save_batch_results(self, batch_num: int, batch_results: List[Dict],
                            entity_records: Optional[List[Dict]] = None):
        """Save batch results to CSV."""
        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            # Remove raw per-entity counts from main feature tables
            count_cols = [c for c in batch_df.columns if c.startswith('count_')]
            if count_cols:
                batch_df = batch_df.drop(columns=count_cols)
            batch_file = self.output_dir / f"batch_{batch_num:04d}.csv"
            try:
                batch_df.to_csv(batch_file, index=False)
                print(f"📊 Batch {batch_num} saved to {batch_file}")
            except Exception as e:
                print(f"⚠️  Failed to save batch {batch_num}: {e}")
        
        self._save_entity_counts(batch_num, entity_records or [])
    
    def load_data(self) -> pd.DataFrame:
        """
        Load transcript data from database.
        
        Returns:
            DataFrame with video_id and caption_text
        """
        print("\n" + "=" * 70)
        print("📥 LOADING DATA FROM DATABASE")
        print("=" * 70)
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT video_id, caption_text FROM CaptionedVideos"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Filter out already processed
            if self.processed_ids:
                df = df[~df['video_id'].isin(self.processed_ids)]
            
            print(f"✅ Loaded {len(df)} transcripts")
            if self.processed_ids:
                print(f"   ({len(self.processed_ids)} already processed, skipped)")
            
            return df
        
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            raise
    
    def process_single(self, video_id: str, text: str, 
                      include_medical: bool = True,
                      include_syntactic: bool = True) -> Optional[Dict]:
        """
        Process a single transcript.
        
        Args:
            video_id: Video ID
            text: Transcript text
            include_medical: Extract medical features
            include_syntactic: Extract syntactic features
        
        Returns:
            Dictionary of features or None if failed
        """
        try:
            # Validate text
            if not text or not isinstance(text, str):
                return None
            
            if len(text.strip()) < 10:
                return None
            
            # Extract features
            features = self.extractor.extract_all_features(
                text,
                include_medical=include_medical,
                include_syntactic=include_syntactic
            )
            
            # Add metadata
            features['video_id'] = video_id
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            return features
        
        except Exception as e:
            print(f"⚠️  Error processing {video_id}: {str(e)[:100]}")
            return None
    
    def process_batch(self, batch_df: pd.DataFrame,
                     include_medical: bool = True,
                     include_syntactic: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a batch of transcripts.
        
        Args:
            batch_df: DataFrame with video_id and caption_text
            include_medical: Extract medical features
            include_syntactic: Extract syntactic features
        
        Returns:
            Tuple of (feature dicts, entity count dicts)
        """
        batch_results = []
        entity_records = []
        
        for idx, row in batch_df.iterrows():
            video_id = row['video_id']
            text = row['caption_text']
            
            result = self.process_single(
                video_id, 
                text,
                include_medical=include_medical,
                include_syntactic=include_syntactic
            )
            
            if result:
                entity_counts = result.pop('entity_group_counts', {})
                entity_records.append(self._build_entity_record(video_id, entity_counts))
                batch_results.append(result)
                self.processed_ids.add(video_id)
            else:
                self.failed_ids[video_id] = "Processing failed"
        
        return batch_results, entity_records
    
    def run(self, 
            include_medical: bool = True,
            include_syntactic: bool = True,
            max_batches: Optional[int] = None) -> pd.DataFrame:
        """
        Run the complete pipeline.
        
        Args:
            include_medical: Extract medical features
            include_syntactic: Extract syntactic features
            max_batches: Maximum number of batches to process (for testing)
        
        Returns:
            DataFrame with all extracted features
        """
        print("\n" + "=" * 70)
        print("🔄 STARTING FEATURE EXTRACTION")
        print("=" * 70)
        
        # Load data
        df = self.load_data()
        total_videos = len(df)
        
        if total_videos == 0:
            print("✅ All videos already processed!")
            return self._load_all_results()
        
        # Calculate batches
        num_batches = (total_videos + self.batch_size - 1) // self.batch_size
        if max_batches:
            num_batches = min(num_batches, max_batches)
        
        print(f"\n📋 Processing Plan:")
        print(f"   Total videos: {total_videos}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Number of batches: {num_batches}")
        print(f"   Medical features: {'✓' if include_medical else '✗'}")
        print(f"   Syntactic features: {'✓' if include_syntactic else '✗'}")
        print()
        
        # Process batches
        for batch_num in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_videos)
            
            batch_df = df.iloc[start_idx:end_idx]
            
            # Process batch
            batch_results, batch_entity_records = self.process_batch(
                batch_df,
                include_medical=include_medical,
                include_syntactic=include_syntactic
            )
            
            if batch_results:
                self.results.extend(batch_results)
            if batch_entity_records:
                self.entity_records.extend(batch_entity_records)
            
            # Save batch results
            self._save_batch_results(batch_num + 1, batch_results, batch_entity_records)
            
            # Save checkpoint
            if (batch_num + 1) % self.checkpoint_freq == 0:
                self._save_checkpoint(batch_num + 1)
            
            # Update progress
            self._save_progress(
                total=total_videos,
                processed=len(self.processed_ids),
                failed=len(self.failed_ids)
            )
        
        # Final checkpoint
        self._save_checkpoint(num_batches)
        
        # Compile final results
        print("\n" + "=" * 70)
        print("📊 COMPILING FINAL RESULTS")
        print("=" * 70)
        
        final_df = self._compile_final_results()
        
        return final_df
    
    def _load_all_results(self) -> pd.DataFrame:
        """Load all batch results from CSV files (excluding entity_counts files)."""
        # Only load batch files that don't contain "entity_counts" in the name
        batch_files = sorted([
            f for f in self.output_dir.glob("batch_*.csv")
            if "_entity_counts" not in f.name
        ])
        
        if not batch_files:
            return pd.DataFrame()
        
        dfs = []
        for file in batch_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️  Failed to load {file}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Remove duplicates by video_id (keep last occurrence)
            if 'video_id' in combined_df.columns:
                initial_count = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['video_id'], keep='last')
                duplicates_removed = initial_count - len(combined_df)
                if duplicates_removed > 0:
                    print(f"🔍 Removed {duplicates_removed} duplicate records by video_id")
            return combined_df
        return pd.DataFrame()
    
    def _compile_final_results(self) -> pd.DataFrame:
        """Compile all results into final DataFrame."""
        # Load all batch results
        final_df = self._load_all_results()
        count_cols = [c for c in final_df.columns if c.startswith('count_')]
        if count_cols:
            final_df = final_df.drop(columns=count_cols)
        
        # Determine data directory (same directory as database)
        db_path = Path(self.db_path)
        data_dir = db_path.parent if db_path.is_file() else db_path
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Save final combined file to data directory
        final_file = data_dir / "transcript_feature_complete.csv"
        try:
            final_df.to_csv(final_file, index=False)
            print(f"✅ Final results saved to: {final_file}")
        except Exception as e:
            print(f"⚠️  Failed to save final results: {e}")
        
        # Save combined entity counts
        entity_df = self._compile_entity_results()
    
        # Print statistics
        print(f"\n📈 EXTRACTION STATISTICS:")
        print(f"   Total processed: {len(self.processed_ids)}")
        print(f"   Successful: {len(final_df)}")
        print(f"   Failed: {len(self.failed_ids)}")
        
        if self.failed_ids:
            failed_file = self.output_dir / "failed_ids.json"
            try:
                with open(failed_file, 'w') as f:
                    json.dump(self.failed_ids, f, indent=2)
                print(f"   Failed IDs saved to: {failed_file}")
            except Exception as e:
                print(f"⚠️  Failed to save failed IDs: {e}")
        
        return final_df
    
    def _compile_entity_results(self) -> pd.DataFrame:
        """Compile per-video entity count data into a combined CSV."""
        if not self.entity_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.entity_records).fillna(0)
        entity_file = self.output_dir / "entity_counts_complete.csv"
        try:
            df.to_csv(entity_file, index=False)
            print(f"🧾 Entity count summary saved to: {entity_file}")
        except Exception as e:
            print(f"⚠️  Failed to save entity count summary: {e}")
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for extracted features.
        
        Args:
            df: DataFrame with extracted features
        
        Returns:
            Dictionary of summary statistics
        """
        if df.empty:
            return {}
        
        # Identify feature columns (exclude metadata)
        metadata_cols = ['video_id', 'text_length', 'word_count']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        stats = {}
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
        
        # Save statistics
        stats_file = self.output_dir / "feature_statistics.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Feature statistics saved to: {stats_file}")
        except Exception as e:
            print(f"⚠️  Failed to save statistics: {e}")
        
        return stats


def find_database(start_path: str = None) -> str:
    """
    Find youtube_data.db by searching current directory, subdirectories, and parent directories.
    
    Args:
        start_path: Starting directory (default: current working directory)
    
    Returns:
        Path to database file
    """
    if start_path is None:
        start_path = os.getcwd()
    
    current = Path(start_path).resolve()
    
    # First, search in current directory and common subdirectories
    search_paths = [
        current / "youtube_data.db",
        current / "data" / "youtube_data.db",
        current / "youtube_data" / "youtube_data.db",
    ]
    
    for db_path in search_paths:
        if db_path.exists():
            return str(db_path)
    
    # Then search up the directory tree
    for _ in range(5):  # Search up to 5 levels
        db_path = current / "youtube_data.db"
        if db_path.exists():
            return str(db_path)
        # Also check data subdirectory in parent
        db_path = current / "data" / "youtube_data.db"
        if db_path.exists():
            return str(db_path)
        current = current.parent
    
    raise FileNotFoundError(
        "youtube_data.db not found. Please specify the path manually."
    )


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("🎬 YOUTUBE TRANSCRIPT FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    
    #Using excel file to get the database path
    
    try:
        db_path = find_database()
        print(f"✅ Found database: {db_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Create pipeline
    pipeline = FeatureExtractionPipeline(
        db_path=db_path,
        output_dir="./feature_extraction_output",
        batch_size=50,  # Process 50 videos per batch
        checkpoint_freq=5,  # Save checkpoint every 5 batches
        load_medical_ner=True,  # Set to False to skip medical features (faster)
        load_spacy=True  # Set to False to skip syntactic features (faster)
    )
    
    # Run pipeline
    results_df = pipeline.run(
        include_medical=True,  # Extract medical features
        include_syntactic=True, # Extract syntactic features
        max_batches=None  # Process all batches (set to 5 for testing)
    )
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("📊 GENERATING SUMMARY STATISTICS")
    print("=" * 70)
    stats = pipeline.get_summary_stats(results_df)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output directory: {pipeline.output_dir}")
    print(f"📄 Main results file: features_complete.csv")
    print(f"🧾 Entity counts file: entity_counts_complete.csv")
    print(f"📄 Statistics file: feature_statistics.json")
    print(f"📄 Progress file: progress.json")
    print(f"💾 Checkpoint file: checkpoint.pkl")
    
    return results_df


if __name__ == "__main__":
    # results = main()
    
