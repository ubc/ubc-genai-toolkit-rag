import { QdrantClient } from '@qdrant/js-client-rest';
import type { Schemas as QdrantSchemas } from '@qdrant/js-client-rest';
import { EmbeddingsModule } from 'ubc-genai-toolkit-embeddings';
import { LoggerInterface } from 'ubc-genai-toolkit-core';
import {
	RAGProviderInterface,
	QdrantConfig,
	RetrievedChunk,
	RetrievalOptions,
} from '../types';
import { v4 as uuidv4 } from 'uuid';

// Simple chunking function (can be replaced with more sophisticated methods later)
function simpleChunker(text: string, chunkSize = 300, overlap = 50): string[] {
	const chunks: string[] = [];
	if (text.length <= chunkSize) {
		return [text];
	}

	let i = 0;
	while (i < text.length) {
		const end = Math.min(i + chunkSize, text.length);
		chunks.push(text.substring(i, end));
		i += chunkSize - overlap;
		if (end === text.length) break; // Exit if we reached the end
	}
	return chunks;
}

export class QdrantProvider implements RAGProviderInterface {
	private client: QdrantClient;
	private config: QdrantConfig;
	private embeddings: EmbeddingsModule;
	private logger: LoggerInterface;
	private isDebug: boolean;

	constructor(
		config: QdrantConfig,
		embeddingsModule: EmbeddingsModule,
		logger: LoggerInterface,
		debug = false
	) {
		this.config = config;
		this.embeddings = embeddingsModule;
		this.logger = logger;
		this.isDebug = debug;

		// Initialize Qdrant client
		this.client = new QdrantClient({
			url: this.config.url,
			apiKey: this.config.apiKey,
		});

		if (this.isDebug) {
			this.logger.debug('QdrantProvider configured:', { url: this.config.url, collectionName: this.config.collectionName });
		}
	}

	async initialize(): Promise<void> {
		try {
			this.logger.info(`Checking for Qdrant collection '${this.config.collectionName}'...`);
			const collections = await this.client.getCollections();
			const collectionExists = collections.collections.some(
				(col) => col.name === this.config.collectionName
			);

			if (!collectionExists) {
				this.logger.warn(
					`Collection '${this.config.collectionName}' not found. Attempting to create...`
				);
				await this.client.createCollection(this.config.collectionName, {
					vectors: {
						size: this.config.vectorSize,
						distance: this.config.distanceMetric,
					},
				});
				this.logger.info(`Collection '${this.config.collectionName}' created successfully.`);
			} else {
				this.logger.info(`Collection '${this.config.collectionName}' exists.`);
				// Optionally, we could verify if the existing collection's config matches
			}
		} catch (error) {
			this.logger.error('Error during Qdrant initialization:', { error });
			throw new Error(`Qdrant initialization failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	async addDocument(content: string, metadata: Record<string, any> = {}): Promise<string[]> {
		this.logger.debug(`Adding document with metadata:`, metadata);
		// 1. Chunk the document
		const chunks = simpleChunker(content);
		this.logger.debug(`Document split into ${chunks.length} chunks.`);
		if (chunks.length === 0) {
			this.logger.warn('Document content resulted in zero chunks. Nothing to add.');
			return [];
		}

		// 2. Get embeddings for all chunks in one batch
		this.logger.debug(`Generating embeddings for ${chunks.length} chunks...`);
		const embeddings = await this.embeddings.embed(chunks);
		this.logger.info(`Successfully generated embeddings for ${embeddings.length}/${chunks.length} chunks.`);

		// 3. Create points, filtering out any that failed to embed
		const points: QdrantSchemas['PointStruct'][] = [];
		const addedChunkIds: string[] = [];
		for (let i = 0; i < chunks.length; i++) {
			const embedding = embeddings[i];
			if (!embedding) {
				this.logger.warn(`Skipping chunk ${i} as it failed to produce an embedding.`);
				continue;
			}
			const chunkId = uuidv4();
			addedChunkIds.push(chunkId);
			points.push({
				id: chunkId,
				vector: embedding,
				payload: {
					...metadata,
					content: chunks[i],
					chunkIndex: i,
				},
			});
		}

		if (points.length === 0) {
			this.logger.warn('All chunks failed to produce embeddings. Nothing to upsert.');
			return [];
		}

		// 4. Upsert points to Qdrant
		try {
			this.logger.debug(`Upserting ${points.length} points to collection '${this.config.collectionName}'...`);
			await this.client.upsert(this.config.collectionName, {
				wait: true,
				points: points,
			});
			this.logger.info(`Successfully upserted ${points.length} points.`);
			return addedChunkIds;
		} catch (error) {
			this.logger.error('Error upserting points to Qdrant:', { error: JSON.stringify(error, null, 2) });
			throw new Error(`Qdrant upsert failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	async retrieveContext(queryText: string, options: RetrievalOptions = {}): Promise<RetrievedChunk[]> {
		this.logger.debug(`Retrieving context for query: "${queryText.substring(0, 50)}..." with options:`, options);
		// 1. Get query embedding
		const [queryVector] = await this.embeddings.embed(queryText);
		if (!queryVector) {
			throw new Error('Failed to generate embedding for the query text.');
		}

		// 2. Search Qdrant
		try {
			const searchResult = await this.client.search(this.config.collectionName, {
				vector: queryVector,
				limit: options.limit ?? 5, // Use provided limit or default
				score_threshold: options.scoreThreshold, // Use provided threshold if any
				// filter: options.filter, // TODO: Add filter conversion if needed
				with_payload: true, // Crucial to get the content back
				with_vector: false, // Usually don't need the vector itself back
			});

			this.logger.debug(`Qdrant search returned ${searchResult.length} results.`);

			// 3. Map results to RetrievedChunk format
			const retrievedChunks: RetrievedChunk[] = searchResult.map((point) => {
				const payload = point.payload as Record<string, any> | undefined;
				const content = payload?.content as string ?? ''; // Extract content
				// Prepare metadata, excluding the content field itself
				const metadata = { ...payload };
				delete metadata.content;

				return {
					content: content,
					score: point.score,
					metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
				};
			});

			return retrievedChunks;
		} catch (error) {
			this.logger.error('Error searching Qdrant:', { error });
			throw new Error(`Qdrant search failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	async getDocumentsByMetadata(filter: Record<string, any>): Promise<any[]> {
		this.logger.debug('Getting all documents by metadata filter:', { filter });

		const allPoints: QdrantSchemas['Record'][] = [];
		let nextPageOffset: QdrantSchemas['ScrollRequest']['offset'] | null | undefined = undefined;

		try {
			do {
				const response = await this.client.scroll(this.config.collectionName, {
					filter: {
						must: Object.entries(filter).map(([key, value]) => ({
							key: key,
							match: { value: value },
						})),
					},
					limit: 250, // Sensible page size
					offset: nextPageOffset,
					with_payload: true,
					with_vector: true,
				});

				if (response.points.length > 0) {
					allPoints.push(...response.points);
				}
				nextPageOffset = response.next_page_offset;

			} while (nextPageOffset);

			this.logger.debug(`Retrieved a total of ${allPoints.length} points for the filter.`);

			return allPoints.map((point) => ({
				id: point.id,
				content: point.payload?.content,
				metadata: point.payload,
				vector: point.vector,
			}));

		} catch (error) {
			this.logger.error('Error scrolling through Qdrant points:', { error });
			throw new Error(`Qdrant scroll failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	async deleteDocumentsByIds(ids: string[]): Promise<void> {
		if (!ids || ids.length === 0) {
			this.logger.warn('No IDs provided for deletion.');
			return;
		}
		this.logger.info(`Attempting to delete ${ids.length} documents by ID from collection '${this.config.collectionName}'.`);
		try {
			await this.client.delete(this.config.collectionName, {
				points: ids,
				wait: true, // Wait for operation to complete
			});
			this.logger.info(`Successfully deleted ${ids.length} documents by ID.`);
		} catch (error) {
			this.logger.error(`Error deleting documents by ID from Qdrant collection '${this.config.collectionName}':`, { error });
			throw new Error(`Qdrant deletion by ID failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	async deleteDocumentsByMetadata(filter: Record<string, any>): Promise<void> {
		if (!filter || Object.keys(filter).length === 0) {
			this.logger.warn('No filter provided for deletion by metadata.');
			return;
		}
		this.logger.info(`Attempting to delete documents matching filter from collection '${this.config.collectionName}':`, { filter });

		// Convert simple key-value filter to Qdrant filter structure
		// This assumes a logical AND ('must') for all conditions
		const conditions: QdrantSchemas['Condition'][] = Object.entries(filter).map(
			([key, value]) => ({
				key: key,
				match: {
					// Qdrant 'match' works for keyword, integer, bool.
					// Might need refinement for text matching or other types.
					value: value,
				},
			})
		);

		// Create the filter object, using type assertion to bypass strict check
		const qdrantFilter = {
			must: conditions,
		} as any as QdrantSchemas['Filter']; // Assert type

		try {
			// Pass the asserted filter object
			await this.client.delete(this.config.collectionName, {
				filter: qdrantFilter,
				wait: true,
			});
			this.logger.info(`Successfully submitted deletion request for documents matching filter.`);
			// Note: Qdrant deletion by filter is async internally, 'wait:true' ensures the operation is queued.
			// We don't get a direct count of deleted items here.
		} catch (error) {
			this.logger.error(`Error deleting documents by metadata filter from Qdrant collection '${this.config.collectionName}':`, { error });
			throw new Error(`Qdrant deletion by filter failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	async deleteStorage(): Promise<void> {
		this.logger.warn(`Attempting to delete Qdrant collection '${this.config.collectionName}'...`);
		try {
			const result = await this.client.deleteCollection(this.config.collectionName);
			if (result) {
				this.logger.info(`Successfully deleted collection '${this.config.collectionName}'.`);
			} else {
				// This case might indicate the collection didn't exist, which is not an error here.
				this.logger.warn(`Deletion command for collection '${this.config.collectionName}' returned false. It may not have existed.`);
			}
		} catch (error) {
			this.logger.error(`Error deleting Qdrant collection '${this.config.collectionName}':`, { error });
			// Don't throw if error is "Not Found", otherwise rethrow
			if (error instanceof Error && /not found/i.test(error.message)) {
				this.logger.warn(`Collection '${this.config.collectionName}' was not found during deletion, which is acceptable.`);
			} else {
				throw new Error(`Qdrant collection deletion failed: ${error instanceof Error ? error.message : String(error)}`);
			}
		}
	}
}