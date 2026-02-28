export function formatNumber(value: unknown, digits = 2): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return numeric.toFixed(digits);
}

export function formatTimestamp(value: unknown): string {
  if (typeof value !== "string" || value.trim() === "") {
    return "-";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function toNumericSeries(rows: Array<Record<string, unknown>>, key: string): number[] {
  return rows
    .map((row) => Number(row[key]))
    .filter((value) => Number.isFinite(value));
}

export function sparklinePath(values: number[], width: number, height: number): string {
  if (values.length === 0) {
    return "";
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const ySpan = max - min || 1;
  const xSpan = Math.max(values.length - 1, 1);

  return values
    .map((value, index) => {
      const x = (index / xSpan) * width;
      const y = height - ((value - min) / ySpan) * height;
      const cmd = index === 0 ? "M" : "L";
      return `${cmd}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}
