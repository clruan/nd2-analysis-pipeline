import { useEffect, useState, type ComponentType } from "react";
import type Plotly from "plotly.js";

const loadingStyle = {
  minHeight: 360,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: "100%"
} as const;

type PlotlyChartProps = Record<string, unknown>;

export default function PlotlyChart(props: PlotlyChartProps) {
  const [PlotComponent, setPlotComponent] = useState<ComponentType<PlotlyChartProps> | null>(null);

  useEffect(() => {
    let active = true;
    const loadPlotly = async () => {
      const [{ default: createPlotlyComponent }, plotlyModule] = await Promise.all([
        import("react-plotly.js/factory"),
        import("plotly.js-dist-min")
      ]);
      if (!active) return;
      const plotlyLib = (plotlyModule.default ?? plotlyModule) as typeof Plotly;
      const Plot = createPlotlyComponent(plotlyLib) as ComponentType<PlotlyChartProps>;
      setPlotComponent(() => Plot);
    };
    loadPlotly();
    return () => {
      active = false;
    };
  }, []);

  if (!PlotComponent) {
    return (
      <div style={loadingStyle}>
        Loading charts…
      </div>
    );
  }

  return <PlotComponent {...props} />;
}
