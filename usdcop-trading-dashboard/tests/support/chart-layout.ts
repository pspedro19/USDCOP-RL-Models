/**
 * LAYOUT PARA GRÁFICOS EN JSDOM.
 *
 * jsdom no hace layout: todo elemento mide 0×0 y el `ResizeObserver` mockeado del
 * setup global nunca dispara. `<ResponsiveContainer>` de recharts se queda por
 * tanto en 0×0 y avisa "width(0) and height(0) of chart should be greater than 0"
 * — y, lo importante, el gráfico NO se renderiza: un test que dice cubrir una
 * vista con gráfico estaba cubriendo una vista sin gráfico.
 *
 * Este helper da dimensiones deterministas al contenedor. No mockea recharts: el
 * chart se monta de verdad, solo se le proporciona el layout que el navegador
 * daría y jsdom no.
 *
 * Uso (una vez por archivo de test, antes de los `render`):
 *
 *     import { stubChartLayout } from '../../support/chart-layout';
 *     stubChartLayout();
 */
import { afterAll, beforeAll } from 'vitest';

const WIDTH = 1024;
const HEIGHT = 480;

type Descriptors = Array<[string, PropertyDescriptor | undefined]>;

export function stubChartLayout(width = WIDTH, height = HEIGHT): void {
  let originals: Descriptors = [];
  let originalRect: PropertyDescriptor | undefined;

  beforeAll(() => {
    originals = (['offsetWidth', 'clientWidth'] as const).map((k) => [
      k, Object.getOwnPropertyDescriptor(HTMLElement.prototype, k),
    ]);
    originals.push(
      ...(['offsetHeight', 'clientHeight'] as const).map(
        (k) => [k, Object.getOwnPropertyDescriptor(HTMLElement.prototype, k)] as
          [string, PropertyDescriptor | undefined],
      ),
    );
    originalRect = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype, 'getBoundingClientRect',
    );

    for (const key of ['offsetWidth', 'clientWidth']) {
      Object.defineProperty(HTMLElement.prototype, key, {
        configurable: true, get: () => width,
      });
    }
    for (const key of ['offsetHeight', 'clientHeight']) {
      Object.defineProperty(HTMLElement.prototype, key, {
        configurable: true, get: () => height,
      });
    }
    Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({
        width, height, top: 0, left: 0, right: width, bottom: height,
        x: 0, y: 0, toJSON: () => ({}),
      }),
    });
  });

  afterAll(() => {
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(HTMLElement.prototype, key, descriptor);
      else delete (HTMLElement.prototype as unknown as Record<string, unknown>)[key];
    }
    if (originalRect) {
      Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', originalRect);
    }
  });
}
