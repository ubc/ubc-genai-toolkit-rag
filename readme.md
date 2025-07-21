# UBC GenAI Toolkit - RAG Module

## Overview

This module provides a standardized interface for building Retrieval-Augmented Generation (RAG) systems. It uses the Facade pattern to simplify interactions with vector stores, such as Qdrant, allowing you to easily store, manage, and retrieve documents for use in generative AI applications.

The module handles the complexities of connecting to a vector database, creating collections, and upserting document embeddings and metadata.

## Installation

```bash
npm install ubc-genai-toolkit-rag ubc-genai-toolkit-core ubc-genai-toolkit-embeddings
```

## Core Concepts

-   **`RAGModule`**: The main class and entry point for interacting with a vector store.
-   **`VectorStore`**: An interface representing the operations of a vector database (e.g., Qdrant).
-   **`RAGConfig`**: The configuration object for the `RAGModule`, specifying the provider and its connection details.
-   **`Document`**: A standardized format for text data, including content and optional metadata.

## Configuration

The `RAGModule` is configured during instantiation with a `RAGConfig` object. The primary supported provider is `qdrant`.

```typescript
import { RAGModule, RAGConfig } from 'ubc-genai-toolkit-rag';
import { ConsoleLogger } from 'ubc-genai-toolkit-core'; // Example logger

// Configuration for the Qdrant provider
const ragConfig: RAGConfig = {
	provider: 'qdrant',
	qdrant: {
		url: process.env.QDRANT_URL || 'http://localhost:6333',
		collectionName: 'my-test-collection',
		// apiKey: process.env.QDRANT_API_KEY, // Optional
	},
	logger: new ConsoleLogger(),
};

// Instantiate the module
const ragModule = new RAGModule(ragConfig);
```

## Usage Examples

### Initialization

First, you need an `EmbeddingsModule` to generate the vectors for your documents, and then you initialize the `RAGModule`.

```typescript
import { EmbeddingsModule } from 'ubc-genai-toolkit-embeddings';
import { RAGModule, RAGConfig } from 'ubc-genai-toolkit-rag';
import { ConsoleLogger } from 'ubc-genai-toolkit-core';

// 1. Configure and create the embeddings module
const embeddings = new EmbeddingsModule({
	// Using fastembed for local embeddings
	providerType: 'fastembed',
	logger: new ConsoleLogger(),
});

// 2. Configure and create the RAG module
const ragConfig: RAGConfig = {
	provider: 'qdrant',
	qdrant: {
		url: 'http://localhost:6333',
		collectionName: 'my-test-collection',
	},
	logger: new ConsoleLogger(),
};
const ragModule = new RAGModule(ragConfig);
```

### Adding Documents to the Vector Store

You can add one or more documents. The `RAGModule` will automatically use the `EmbeddingsModule` to create vectors before storing them.

```typescript
async function addDocuments() {
	const documents = [
		{
			pageContent:
				'UBC is a public research university in British Columbia, Canada.',
		},
		{ pageContent: 'It was established in 1908.' },
		{
			pageContent:
				'The Vancouver campus is located on the traditional, ancestral, and unceded territory of the Musqueam people.',
		},
	];

	console.log('Adding documents to the vector store...');
	await ragModule.addDocuments(documents, embeddings);
	console.log('Documents added successfully.');
}
```

### Querying for Similar Documents

Use a query string to find the most relevant documents in the vector store.

```typescript
async function findSimilarDocuments(query: string) {
	console.log(`Querying for documents similar to: "${query}"`);

	const results = await ragModule.query(query, embeddings, {
		topK: 2, // Find the top 2 most similar documents
	});

	console.log('Query Results:');
	results.forEach((result, index) => {
		console.log(`  ${index + 1}. Content: "${result.pageContent}"`);
		console.log(`     Score: ${result.score}`);
	});
}

// Example usage
await addDocuments();
await findSimilarDocuments('Tell me about UBC');
```

## Error Handling

The module uses the common error types from `ubc-genai-toolkit-core`.
