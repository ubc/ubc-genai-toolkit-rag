# UBC GenAI Toolkit - RAG Module

## Overview

This module provides a standardized interface for building Retrieval-Augmented Generation (RAG) systems. It uses the Facade pattern to simplify interactions with vector stores, such as Qdrant, allowing you to easily store, manage, and retrieve documents for use in generative AI applications.

The module handles the complexities of connecting to a vector database, creating collections, and upserting document embeddings and metadata.

## Installation

```bash
npm install ubc-genai-toolkit-rag ubc-genai-toolkit-core ubc-genai-toolkit-embeddings ubc-genai-toolkit-chunking
```

## Core Concepts

-   **`RAGModule`**: The main class and entry point for interacting with a vector store.
-   **`VectorStore`**: An interface representing the operations of a vector database (e.g., Qdrant).
-   **`RAGConfig`**: The configuration object for the `RAGModule`, specifying the provider, embeddings, and chunking strategy.
-   **`Document`**: A standardized format for text data, including content and optional metadata.

## Configuration

The `RAGModule` is configured by passing a `RAGConfig` object to the static `RAGModule.create()` method. This method asynchronously initializes the module and its dependencies.

A complete configuration requires specifying the vector store provider (e.g., `qdrant`), the embeddings model, and optionally, a chunking strategy.

```typescript
import { RAGModule, RAGConfig } from 'ubc-genai-toolkit-rag';
import { ConsoleLogger } from 'ubc-genai-toolkit-core';

// Example complete configuration for the RAGModule
const ragConfig: RAGConfig = {
	provider: 'qdrant',
	qdrantConfig: {
		url: process.env.QDRANT_URL || 'http://localhost:6333',
		collectionName: 'my-test-collection',
		vectorSize: 384, // Dimensionality of the embeddings model
		distanceMetric: 'Cosine',
	},
	embeddingsConfig: {
		provider: 'fastembed',
		model: 'bge-small-en-v1.5',
	},
	// Optional: Add chunking configuration (see below)
	// chunkingConfig: { ... }

	logger: new ConsoleLogger(),
	debug: true,
};

// Asynchronously create and initialize the module
const ragModule = await RAGModule.create(ragConfig);
```

### Chunking Configuration

When adding documents, the `RAGModule` automatically splits them into smaller chunks. You can control this behavior using the `chunkingConfig` property.

#### 1. Default Behavior (No Configuration)

If you do not provide a `chunkingConfig`, the module uses a basic, internal chunker that splits text by a fixed character length. This is suitable for quick tests but is not recommended for production.

#### 2. Using the `ChunkingModule`

For more advanced chunking, you can leverage the `ubc-genai-toolkit-chunking` module. This gives you access to multiple strategies like `recursiveCharacter` or `token`-based splitting.

```typescript
import { RAGConfig } from 'ubc-genai-toolkit-rag';

const ragConfig: RAGConfig = {
	// ... other properties (provider, qdrantConfig, etc.)
	chunkingConfig: {
		strategy: 'recursiveCharacter',
		defaultOptions: {
			chunkSize: 500,
			chunkOverlap: 50,
		},
	},
};
```

#### 3. Using a Custom Chunking Function

For maximum flexibility, you can provide your own function to handle chunking. The function should accept a `string` and return an array of `string` chunks.

```typescript
import { RAGConfig } from 'ubc-genai-toolkit-rag';

const myCustomChunker = (text: string): string[] => {
	// Your custom logic to split text into chunks
	return text.split('\n\n'); // Example: split by paragraph
};

const ragConfig: RAGConfig = {
	// ... other properties (provider, qdrantConfig, etc.)
	chunkingConfig: myCustomChunker,
};
```

## Usage Examples

### Initialization

First, configure and create an instance of the `RAGModule`. The module handles the initialization of its internal dependencies, like the embeddings model.

```typescript
import { RAGModule, RAGConfig } from 'ubc-genai-toolkit-rag';
import { ConsoleLogger } from 'ubc-genai-toolkit-core';

async function initializeRagModule() {
	const ragConfig: RAGConfig = {
		provider: 'qdrant',
		qdrantConfig: {
			url: 'http://localhost:6333',
			collectionName: 'my-test-collection',
			vectorSize: 384, // e.g., for bge-small-en-v1.5
			distanceMetric: 'Cosine',
		},
		embeddingsConfig: {
			provider: 'fastembed',
			model: 'bge-small-en-v1.5',
		},
		logger: new ConsoleLogger(),
	};

	const ragModule = await RAGModule.create(ragConfig);
	return ragModule;
}

const ragModule = await initializeRagModule();
```

### Adding Documents to the Vector Store

You can add a document as a single string. The `RAGModule` will automatically handle chunking and embedding before storing the vectorized chunks.

```typescript
async function addDocument(ragModule: RAGModule) {
	const documentContent = `UBC is a public research university in British Columbia, Canada.
It was established in 1908.
The Vancouver campus is located on the traditional, ancestral, and unceded territory of the Musqueam people.`;

	console.log('Adding document to the vector store...');
	const chunkIds = await ragModule.addDocument(documentContent, {
		source: 'ubc-facts.txt',
	});
	console.log(
		`Document added successfully. Chunk IDs: ${chunkIds.join(', ')}`
	);
}
```

### Querying for Similar Documents

Use a query string to find the most relevant document chunks in the vector store.

```typescript
async function findSimilarDocuments(ragModule: RAGModule, query: string) {
	console.log(`Querying for documents similar to: "${query}"`);

	const results = await ragModule.retrieveContext(query, {
		limit: 2, // Find the top 2 most similar document chunks
	});

	console.log('Query Results:');
	results.forEach((result, index) => {
		console.log(`  ${index + 1}. Content: "${result.content}"`);
		console.log(`     Score: ${result.score}`);
		console.log(`     Metadata:`, result.metadata);
	});
}

// Example usage
await addDocument(ragModule);
await findSimilarDocuments(ragModule, 'Tell me about UBC');
```

## Error Handling

The module uses the common error types from `ubc-genai-toolkit-core`.
