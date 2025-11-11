import { useState } from "react";
import { Box, Divider, IconButton, Stack, Tooltip } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import MenuOpenIcon from "@mui/icons-material/MenuOpen";
import LeftPanel from "./components/LeftPanel";
import AnalysisBoard from "./components/AnalysisBoard";
import PreviewPane from "./components/PreviewPane";

export default function App() {
  const [leftOpen, setLeftOpen] = useState(true);

  return (
    <Stack direction="row" spacing={0} sx={{ minHeight: "100vh", bgcolor: "background.default", color: "text.primary" }}>
      <Box
        sx={{
          width: leftOpen ? 360 : 0,
          display: leftOpen ? "block" : "none",
          borderRight: "1px solid rgba(15,23,42,0.08)",
          backgroundColor: "background.paper",
          boxShadow: "4px 0 24px rgba(15,23,42,0.04)",
          overflowY: "auto",
          transition: "width 0.25s ease"
        }}
      >
        <LeftPanel />
      </Box>
      <Box
        sx={{
          width: 44,
          flexShrink: 0,
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          pt: 1
        }}
      >
        <Tooltip title={leftOpen ? "Hide controls" : "Show controls"}>
          <IconButton size="small" onClick={() => setLeftOpen((open) => !open)}>
            {leftOpen ? <MenuOpenIcon fontSize="small" /> : <MenuIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
      </Box>
      <Box sx={{ flex: 2, minWidth: 0, overflowY: "auto", p: 3 }}>
        <AnalysisBoard />
      </Box>
      <Divider orientation="vertical" flexItem sx={{ borderColor: "rgba(15,23,42,0.08)" }} />
      <Box
        sx={{
          flex: 1.6,
          minWidth: 480,
          borderLeft: "1px solid rgba(15,23,42,0.08)",
          backgroundColor: "background.paper",
          boxShadow: "-4px 0 24px rgba(15,23,42,0.04)",
          overflowY: "auto",
          p: 2
        }}
      >
        <PreviewPane />
      </Box>
    </Stack>
  );
}
