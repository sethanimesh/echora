export type Point = { x: number; y: number };
export type Calibration = { x: number[]; y: number[] };
export function solve3(matrix: number[][], values: number[]): number[] | null {
  const a = matrix.map((row, i) => [...row, values[i]]);
  for (let c = 0; c < 3; c++) {
    let pivot = c;
    for (let r = c + 1; r < 3; r++)
      if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    if (Math.abs(a[pivot][c]) < 1e-9) return null;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    const scale = a[c][c];
    for (let k = c; k < 4; k++) a[c][k] /= scale;
    for (let r = 0; r < 3; r++)
      if (r !== c) {
        const f = a[r][c];
        for (let k = c; k < 4; k++) a[r][k] -= f * a[c][k];
      }
  }
  return a.map((row) => row[3]);
}
export function fitCalibration(
  samples: { feature: Point; target: Point }[],
): Calibration | null {
  const matrix = Array.from({ length: 3 }, () => [0, 0, 0]),
    x = [0, 0, 0],
    y = [0, 0, 0];
  for (const { feature, target } of samples) {
    const row = [feature.x, feature.y, 1];
    for (let i = 0; i < 3; i++) {
      x[i] += row[i] * target.x;
      y[i] += row[i] * target.y;
      for (let j = 0; j < 3; j++) matrix[i][j] += row[i] * row[j];
    }
  }
  const cx = solve3(matrix, x),
    cy = solve3(matrix, y);
  if (!cx || !cy) return null;
  const result = { x: cx, y: cy };
  const error =
    samples.reduce((sum, s) => {
      const p = mapPoint(result, s.feature);
      return sum + Math.hypot(p.x - s.target.x, p.y - s.target.y);
    }, 0) / samples.length;
  return Number.isFinite(error) && error < 0.14 ? result : null;
}
export function mapPoint(c: Calibration, p: Point): Point {
  return {
    x: c.x[0] * p.x + c.x[1] * p.y + c.x[2],
    y: c.y[0] * p.x + c.y[1] * p.y + c.y[2],
  };
}
export function faceFeature(
  points: Point[],
  mode: 'head' | 'gaze',
): Point | null {
  if (points.length < 478) return null;
  if (mode === 'head') return { x: points[1].x, y: points[1].y };
  function eye(
    iris: number,
    left: number,
    right: number,
    top: number,
    bottom: number,
  ) {
    const width = points[right].x - points[left].x,
      height = points[bottom].y - points[top].y;
    if (Math.abs(width) < 0.01 || Math.abs(height / width) < 0.1) return null;
    return {
      x: (points[iris].x - points[left].x) / width,
      y: (points[iris].y - points[top].y) / Math.abs(width),
    };
  }
  const a = eye(468, 33, 133, 159, 145),
    b = eye(473, 362, 263, 386, 374);
  return a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } : null;
}
