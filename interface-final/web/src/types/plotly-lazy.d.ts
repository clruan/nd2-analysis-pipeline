declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  import type Plotly from "plotly.js";

  export default function createPlotlyComponent(plotly: typeof Plotly): ComponentType<Record<string, unknown>>;
}

declare module "plotly.js-dist-min" {
  import type Plotly from "plotly.js";
  const PlotlyLib: typeof Plotly;
  export default PlotlyLib;
}
