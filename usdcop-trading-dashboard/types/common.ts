/**
 * Common Utility Types
 * ====================
 *
 * Tipos reutilizables para todo el proyecto
 */

// === UTILITY TYPES ===

/**
 * Hace todos los campos de un tipo nullable
 */
export type Nullable<T> = T | null;

/**
 * Hace todos los campos de un tipo opcionales y nullable
 */
export type Optional<T> = Partial<T> | null;

/**
 * Estado de una operación asíncrona
 */
export interface AsyncState<T> {
  data: Nullable<T>;
  loading: boolean;
  error: Nullable<Error | string>;
  timestamp: Nullable<number>;
}

/**
 * Estado de carga simple
 */
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

/**
 * Respuesta estándar de API
 */
export interface ApiResponse<T> {
  success: boolean;
  data: Nullable<T>;
  message?: string;
  error?: string;
  timestamp?: string;
  metadata?: Record<string, any>;
}

/**
 * Respuesta paginada de API
 */
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

/**
 * Parámetros de paginación
 */
export interface PaginationParams {
  page?: number;
  limit?: number;
  offset?: number;
}

/**
 * Parámetros de ordenamiento
 */
export interface SortParams {
  field: string;
  order: 'asc' | 'desc';
}

/**
 * Parámetros de filtro
 */
export interface FilterParams {
  [key: string]: string | number | boolean | null | undefined;
}

/**
 * Rango de fechas
 */
export interface DateRange {
  startDate: Date | string;
  endDate: Date | string;
}

/**
 * Rango de tiempo con timestamps
 */
export interface TimeRange {
  start: number;
  end: number;
}

/**
 * Coordenadas 2D
 */
export interface Coordinates {
  x: number;
  y: number;
}

/**
 * Dimensiones
 */
export interface Dimensions {
  width: number;
  height: number;
}

/**
 * Bounding box
 */
export interface BoundingBox extends Coordinates, Dimensions {}

/**
 * Color con opacidad
 */
export interface Color {
  r: number;
  g: number;
  b: number;
  a?: number;
}

/**
 * Estado de validación
 */
export interface ValidationState {
  isValid: boolean;
  errors: string[];
  warnings?: string[];
}

/**
 * Metadatos genéricos
 */
export interface Metadata {
  createdAt?: string;
  updatedAt?: string;
  createdBy?: string;
  updatedBy?: string;
  version?: number;
  tags?: string[];
  [key: string]: string | number | boolean | null | undefined | string[];
}

/**
 * Resultado con éxito/fallo
 */
export type Result<T, E = Error> =
  | { success: true; value: T }
  | { success: false; error: E };

/**
 * Función de callback genérica
 */
export type Callback<T = void> = (value: T) => void;

/**
 * Función de callback asíncrona
 */
export type AsyncCallback<T = void> = (value: T) => Promise<void>;

/**
 * Función de predicado
 */
export type Predicate<T> = (value: T) => boolean;

/**
 * Función de transformación
 */
export type Transformer<T, U> = (value: T) => U;

/**
 * Objeto con claves string y valores de tipo T
 */
export type Dictionary<T> = Record<string, T>;

/**
 * Deep Partial - hace todos los campos anidados opcionales
 */
export type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>;
} : T;

/**
 * Deep Readonly - hace todos los campos anidados readonly
 */
export type DeepReadonly<T> = T extends object ? {
  readonly [P in keyof T]: DeepReadonly<T[P]>;
} : T;

/**
 * Extrae el tipo de un array
 */
export type ArrayElement<T> = T extends (infer U)[] ? U : never;

/**
 * Extrae el tipo de una promesa
 */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/**
 * Tipo que puede ser un valor o un array de valores
 */
export type OneOrMany<T> = T | T[];

/**
 * Tipo que puede ser un valor o una función que retorna ese valor
 */
export type ValueOrFactory<T> = T | (() => T);
