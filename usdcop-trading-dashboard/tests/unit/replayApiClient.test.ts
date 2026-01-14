// tests/replayApiClient.test.ts
import { config } from '../../lib/replayApiClient';

test('modelId comes from environment', () => {
    process.env.CURRENT_MODEL_ID = 'test_model_id';
    expect(config.modelId).not.toBe('v19-checkpoint');
    expect(config.modelId).toBe(process.env.CURRENT_MODEL_ID);
});
