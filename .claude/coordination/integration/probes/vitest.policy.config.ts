import path from 'node:path';

const projectRoot = path.resolve(__dirname, '../../../..');
const dashboardRoot = path.join(projectRoot, 'usdcop-trading-dashboard');

export default {
  root: projectRoot,
  test: {
    environment: 'node',
    include: [
      'usdcop-trading-dashboard/tests/unit/contracts/policy-backend-parity.test.ts',
      'usdcop-trading-dashboard/tests/unit/contracts/policy-contract-parity.test.ts',
    ],
  },
  resolve: {
    alias: {
      '@': dashboardRoot,
    },
  },
};
