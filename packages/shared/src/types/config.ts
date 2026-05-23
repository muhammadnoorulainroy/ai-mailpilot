/**
 * Shared configuration type definitions, including model preset options used
 * to pair embedding and generation models for the app.
 */

/**
 * Predefined model configuration option pairing an embedding and generation
 * model, with the minimum RAM needed to run them.
 */
export interface ModelPreset {
  id: 'lightweight' | 'recommended' | 'institutional' | 'maximum' | 'custom';
  label: string;
  description: string;
  embeddingModel: string;
  generationModel: string;
  minRamGb: number;
}
