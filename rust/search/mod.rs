// Search module - Core search and retrieval functionality

pub mod centroid_selector;
pub mod decompressor;
pub mod loader;
pub mod merger;
pub mod scorer;
use std::sync::Arc;

// Re-export main types for convenience
pub use centroid_selector::CentroidSelector;
pub use decompressor::CentroidDecompressor;
pub use loader::IndexLoader;
pub use merger::ResultMerger;
pub use scorer::{WARPScorer, ShardedWARPScorer};

use anyhow::Result;

use crate::utils::types::{Query, ReadOnlyIndex, SearchConfig, SearchResult, ShardedIndex};

/// Main search interface combining all components
pub struct Searcher {
    scorer: WARPScorer,
}

impl Searcher {
    /// Create a new searcher with loaded index
    pub fn new(index: &Arc<ReadOnlyIndex>, config: &SearchConfig) -> Result<Self> {
        let scorer = WARPScorer::new(index, config.clone())?;
        Ok(Self { scorer })
    }

    /// Search for top-k passages given a query
    pub fn search(&self, query: Query) -> Result<Vec<SearchResult>> {
        self.scorer.rank(&query)
    }
}

/// Sharded search interface distributing work across multiple GPUs.
pub struct ShardedSearcherImpl {
    scorer: ShardedWARPScorer,
}

impl ShardedSearcherImpl {
    pub fn new(sharded_index: ShardedIndex, config: &SearchConfig) -> Result<Self> {
        let scorer = ShardedWARPScorer::new(sharded_index, config.clone())?;
        Ok(Self { scorer })
    }

    /// Build a searcher for this search call using `config` (e.g. auto-tuned hyperparams).
    pub fn new_ref(sharded_index: &ShardedIndex, config: &SearchConfig) -> Result<Self> {
        Self::new(sharded_index.clone(), config)
    }

    pub fn search(&self, query: Query) -> Result<Vec<SearchResult>> {
        self.scorer.rank(&query)
    }
}
