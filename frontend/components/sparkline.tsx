"use client";

import { sparklinePath } from "@/lib/format";

type SparklineProps = {
  values: number[];
  stroke?: string;
};

export function Sparkline({ values, stroke }: SparklineProps) {
  if (values.length < 2) {
    return <p className="muted">Need at least two points.</p>;
  }

  const width = 260;
  const height = 70;
  const path = sparklinePath(values, width, height);

  return (
    <svg className="spark-wrap" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="trend">
      <path d={path} style={stroke ? { stroke } : undefined} />
    </svg>
  );
}
