import path from 'node:path';

const projectRoot = path.resolve(__dirname, '../../../..');
const dashboardRoot = path.join(projectRoot, 'usdcop-trading-dashboard');

export default {
  root: projectRoot,
  test: {
    environment: 'node',
    include: [
      '.claude/coordination/integration/probes/billing_reference_replay_probe.test.ts',
    ],
  },
  resolve: {
    alias: {
      '@': dashboardRoot,
    },
  },
};
