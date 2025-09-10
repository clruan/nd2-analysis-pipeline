import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './index.css';

const API_BASE = 'http://localhost:8000';

function App() {
  // State management
  const [thresholds, setThresholds] = useState({
    channel_1: 2500,
    channel_2: 2500,
    channel_3: 300
  });
  
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [studyLoaded, setStudyLoaded] = useState(false);
  const [fileName, setFileName] = useState('');
  const [studyId, setStudyId] = useState('test_study');
  
  // Statistical analysis state
  const [statisticsEnabled, setStatisticsEnabled] = useState(false);
  const [comparisonMode, setComparisonMode] = useState('all_vs_one');
  const [referenceGroup, setReferenceGroup] = useState('');
  const [comparisonPairs, setComparisonPairs] = useState([]);
  const [testType, setTestType] = useState('auto');
  const [significanceDisplay, setSignificanceDisplay] = useState('stars');
  const [statisticsData, setStatisticsData] = useState(null);
  const [hoveredMouse, setHoveredMouse] = useState(null);
  const [hoverTimeout, setHoverTimeout] = useState(null);

  // Fetch data when thresholds change (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchAnalysis();
      if (statisticsEnabled) {
        fetchStatistics();
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timer);
  }, [thresholds, statisticsEnabled, comparisonMode, referenceGroup, comparisonPairs, testType, significanceDisplay]);

  // Cleanup hover timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeout) {
        clearTimeout(hoverTimeout);
      }
    };
  }, [hoverTimeout]);

  const fetchAnalysis = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE}/api/studies/${studyId}/analyze`,
        thresholds
      );
      setData(response.data);
    } catch (error) {
      console.error('Error fetching analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const statisticsRequest = {
        ...thresholds,
        comparison_mode: comparisonMode,
        reference_group: referenceGroup || undefined,
        comparison_pairs: comparisonPairs.length > 0 ? comparisonPairs : undefined,
        test_type: testType,
        significance_display: significanceDisplay
      };

      const response = await axios.post(
        `${API_BASE}/api/studies/${studyId}/statistics`,
        statisticsRequest
      );
      setStatisticsData(response.data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
      setStatisticsData(null);
    }
  };

  const handleThresholdChange = (channel, value) => {
    setThresholds(prev => ({
      ...prev,
      [channel]: parseInt(value)
    }));
  };

  const loadStudyData = async () => {
    if (!fileName.trim()) {
      alert('Please enter a filename');
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE}/api/studies/load`, {
        file_path: fileName
      });
      
      if (response.data) {
        setStudyLoaded(true);
        setStudyId(response.data.study_id);
        console.log('Study loaded:', response.data);
        // Fetch initial data with correct study ID
        setTimeout(async () => {
          await fetchAnalysis();
        }, 100);
      }
    } catch (error) {
      console.error('Error loading study:', error);
      alert(`Error loading study: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Create interactive boxplots like the original pipeline
  const createBoxplotForMetric = (metric, title, channelKey) => {
    if (!data || !data.mouse_averages) return null;

    const groups = [...new Set(data.mouse_averages.map(d => d.group))];
    const annotations = [];
    const shapes = [];
    
    const traces = groups.map(group => {
      const groupData = data.mouse_averages.filter(d => d.group === group);
      const values = groupData.map(d => d[metric]);
      
      return {
        y: values,
        type: 'box',
        name: group,
        boxpoints: 'all',        // Show all individual points
        jitter: 0.3,             // Spread points horizontally  
        pointpos: 0,             // Position points INSIDE the boxes (0 = center)
        hoveron: 'points',       // Only trigger hover on the actual points, not the box
        marker: {
          size: 12,              // Bigger size for easier access
          opacity: 0.9,          // High opacity for visibility
          line: {                // Add outline to dots
            width: 2,
            color: 'white'       // White outline for contrast
          }
        },
        line: {
          width: 2
        },
        // Add mouse ID as hover text
        text: groupData.map(d => `Mouse: ${d.mouse_id}<br>Value: ${d[metric] !== undefined ? d[metric].toFixed(2) : 'N/A'}%`),
        hovertemplate: '%{text}<extra></extra>',
        hoverinfo: 'text'        // Only show the custom text on hover
      };
    });

    // Add individual sample traces as overlay (no dynamic trace addition)
    if (hoveredMouse && data.individual_images) {
      groups.forEach((group, groupIndex) => {
        const individualSamples = data.individual_images.filter(img => 
          img.mouse_id === hoveredMouse && img.group === group
        );
        
        if (individualSamples.length > 0) {
          traces.push({
            x: individualSamples.map(() => group), // Use group name, not index
            y: individualSamples.map(img => img[metric]),
            type: 'scatter',
            mode: 'markers',
            name: `${hoveredMouse} replicates`,
            marker: {
              size: 8,
              color: 'rgba(100, 100, 100, 0.5)', // Slightly more visible
              symbol: 'circle-open',
              line: {
                width: 2,
                color: 'gray'
              }
            },
            showlegend: false,
            hovertemplate: `Replicate: %{text}<br>Value: %{y:.2f}%<extra></extra>`,
            text: individualSamples.map(img => img.filename),
            hoverinfo: 'text'  // Ensure hover works on replicate dots too
          });
        }
      });
    }

    // Add statistical annotations if available
    if (statisticsEnabled && statisticsData && statisticsData.statistics[channelKey]) {
      const channelStats = statisticsData.statistics[channelKey];
      
      if (channelStats.comparison_mode === "all_vs_one" && channelStats.overall_test) {
        // Add overall ANOVA result
        annotations.push({
          x: 0.5,
          y: 1.02,
          xref: 'paper',
          yref: 'paper',
          text: `Overall: ${channelStats.overall_test.significance}`,
          showarrow: false,
          font: { size: 12, color: 'red' }
        });
      }
      
        // Add pairwise comparison annotations in data layer (above boxplots)
        if (channelStats.pairwise_comparisons) {
          // Find the maximum value across all groups for positioning
          const allValues = data.mouse_averages
            .map(d => d[metric])
            .filter(val => val !== undefined && !isNaN(val));
          const maxValue = allValues.length > 0 ? Math.max(...allValues) : 100;
          
          channelStats.pairwise_comparisons.forEach((comparison, index) => {
            const group1Index = groups.indexOf(comparison.group1);
            const group2Index = groups.indexOf(comparison.group2);
            
            if (group1Index !== -1 && group2Index !== -1) {
              // Position above the highest data point
              const yPos = maxValue * (1.1 + index * 0.15);
              const xCenter = (group1Index + group2Index) / 2;
              
              // Add horizontal comparison bar
              shapes.push({
                type: 'line',
                x0: group1Index,
                y0: yPos,
                x1: group2Index,
                y1: yPos,
                line: {
                  color: 'black',
                  width: 1.5
                }
              });
              
              // Add vertical lines at each end
              shapes.push({
                type: 'line',
                x0: group1Index,
                y0: yPos - maxValue * 0.02,
                x1: group1Index,
                y1: yPos,
                line: {
                  color: 'black',
                  width: 1.5
                }
              });
              
              shapes.push({
                type: 'line',
                x0: group2Index,
                y0: yPos - maxValue * 0.02,
                x1: group2Index,
                y1: yPos,
                line: {
                  color: 'black',
                  width: 1.5
                }
              });
              
              // Add significance text above the bar
              annotations.push({
                x: xCenter,
                y: yPos + maxValue * 0.05,
                text: comparison.significance,
                showarrow: false,
                font: { 
                  size: 16, 
                  color: comparison.p_value < 0.05 ? 'red' : 'black'
                },
                xanchor: 'center',
                yanchor: 'bottom'
              });
            }
          });
        }
    }

    return (
      <Plot
        data={traces}
        layout={{
          title: {
            text: title,
            font: { size: 16 }
          },
          yaxis: { 
            title: 'Positive Area (%)',
            titlefont: { size: 14 }
          },
          xaxis: {
            title: 'Treatment Group',
            titlefont: { size: 14 }
          },
           width: 800,
           height: 500,  // Increased height for annotations
           margin: { t: 120, b: 80, l: 60, r: 40 },  // Extra top margin for publication-style annotations
          showlegend: false,
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          annotations: annotations,
          shapes: shapes
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }}
        onHover={(eventData) => {
          if (eventData.points && eventData.points[0]) {
            const point = eventData.points[0];
            
            // Clear any existing timeout
            if (hoverTimeout) {
              clearTimeout(hoverTimeout);
            }
            
            // Check if hovering over a boxplot point or already hovering over replicates
            if (point.data.type === 'box') {
              const pointIndex = point.pointIndex;
              const traceIndex = point.curveNumber;
              const group = groups[traceIndex];
              const groupData = data.mouse_averages.filter(d => d.group === group);
              
              if (groupData[pointIndex]) {
                setHoveredMouse(groupData[pointIndex].mouse_id);
              }
            } else if (point.data.name && point.data.name.includes('replicates')) {
              // Keep the current hovered mouse when hovering over replicates
              // This prevents disappearing when moving between mouse dot and replicates
              // hoveredMouse stays the same
            }
          }
        }}
        onUnhover={() => {
          // Clear any existing timeout
          if (hoverTimeout) {
            clearTimeout(hoverTimeout);
          }
          
          // Longer delay to give time to move between mouse dot and replicates
          const timeout = setTimeout(() => {
            setHoveredMouse(null);
          }, 200);
          
          setHoverTimeout(timeout);
        }}
      />
    );
  };

  return (
    <div className="app">
      <h1>ND2 Interactive Threshold Analysis</h1>

      <div className="data-load-section">
        <h3>Load Your Study Data</h3>
        <input
          type="text"
          className="file-input"
          placeholder="threshold_results_Study_2_Octapharrma_Lung_20X_vWF_Pselectin_CdD31_05072025.json"
          value={fileName}
          onChange={(e) => setFileName(e.target.value)}
        />
        <button className="load-button" onClick={loadStudyData} disabled={loading}>
          {loading ? 'Loading...' : 'Load Study'}
        </button>
        <p>Status: {studyLoaded ? '✅ Real data loaded' : '🔬 Using mock data'}</p>
      </div>
      
      <div className="threshold-controls">
        <div className="channel-control">
          <label>Channel 1 (Green)</label>
          <input
            type="range"
            min="0"
            max="4095"
            value={thresholds.channel_1}
            onChange={(e) => handleThresholdChange('channel_1', e.target.value)}
          />
          <div className="channel-value channel-1">{thresholds.channel_1}</div>
        </div>
        
        <div className="channel-control">
          <label>Channel 2 (Red)</label>
          <input
            type="range"
            min="0"
            max="4095"
            value={thresholds.channel_2}
            onChange={(e) => handleThresholdChange('channel_2', e.target.value)}
          />
          <div className="channel-value channel-2">{thresholds.channel_2}</div>
        </div>
        
        <div className="channel-control">
          <label>Channel 3 (Blue)</label>
          <input
            type="range"
            min="0"
            max="4095"
            value={thresholds.channel_3}
            onChange={(e) => handleThresholdChange('channel_3', e.target.value)}
          />
          <div className="channel-value channel-3">{thresholds.channel_3}</div>
        </div>
      </div>

      <div className="statistics-controls">
        <h3>Statistical Analysis</h3>
        <div className="stats-toggle">
          <label>
            <input
              type="checkbox"
              checked={statisticsEnabled}
              onChange={(e) => setStatisticsEnabled(e.target.checked)}
            />
            Enable Statistical Analysis
          </label>
        </div>
        
        {statisticsEnabled && (
          <div className="stats-options">
            <div className="stats-row">
              <div className="stats-group">
                <label>Comparison Mode:</label>
                <select value={comparisonMode} onChange={(e) => setComparisonMode(e.target.value)}>
                  <option value="all_vs_one">All vs One Group</option>
                  <option value="pairs">Specified Pairs</option>
                </select>
              </div>
              
              <div className="stats-group">
                <label>Test Type:</label>
                <select value={testType} onChange={(e) => setTestType(e.target.value)}>
                  <option value="auto">Auto-detect</option>
                  <option value="parametric">Parametric (t-test/ANOVA)</option>
                  <option value="non_parametric">Non-parametric (Mann-Whitney/Kruskal-Wallis)</option>
                </select>
              </div>
              
              <div className="stats-group">
                <label>Display:</label>
                <select value={significanceDisplay} onChange={(e) => setSignificanceDisplay(e.target.value)}>
                  <option value="stars">Significance Stars (*)</option>
                  <option value="p_values">P-values</option>
                </select>
              </div>
            </div>
            
            {comparisonMode === 'all_vs_one' && data && data.mouse_averages && (
              <div className="stats-row">
                <div className="stats-group">
                  <label>Reference Group:</label>
                  <select value={referenceGroup} onChange={(e) => setReferenceGroup(e.target.value)}>
                    <option value="">Auto (first group)</option>
                    {[...new Set(data.mouse_averages.map(d => d.group))].map(group => (
                      <option key={group} value={group}>{group}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}
            
            {comparisonMode === 'pairs' && data && data.mouse_averages && (
              <div className="stats-row">
                <div className="stats-group">
                  <label>Comparison Pairs:</label>
                  <div className="pairs-selector">
                    {/* Simplified pairs selection - you can enhance this */}
                    <button onClick={() => {
                      const groups = [...new Set(data.mouse_averages.map(d => d.group))];
                      if (groups.length >= 2) {
                        setComparisonPairs([[groups[0], groups[1]]]);
                      }
                    }}>
                      Add Default Pair
                    </button>
                    <span>{comparisonPairs.length} pairs selected</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="visualization">
        {loading ? (
          <div className="loading">
            <p>Updating analysis...</p>
          </div>
        ) : (
          <div>
            <div className="plot-container">
              <h3>Channel 1 (Green) - All Treatment Groups</h3>
              {createBoxplotForMetric('channel_1_area', 'Channel 1 Area by Group', 'channel_1')}
            </div>
            
            <div className="plot-container">
              <h3>Channel 2 (Red) - All Treatment Groups</h3>
              {createBoxplotForMetric('channel_2_area', 'Channel 2 Area by Group', 'channel_2')}
            </div>
            
            <div className="plot-container">
              <h3>Channel 3 (Blue) - All Treatment Groups</h3>
              {createBoxplotForMetric('channel_3_area', 'Channel 3 Area by Group', 'channel_3')}
            </div>
            
            <div className="plot-container">
              <h3>Green/Blue Ratio - All Treatment Groups</h3>
              {createBoxplotForMetric('channel_1_3_ratio', 'Channel 1/3 Ratio by Group', 'channel_1_3_ratio')}
            </div>
            
            <div className="plot-container">
              <h3>Red/Blue Ratio - All Treatment Groups</h3>
              {createBoxplotForMetric('channel_2_3_ratio', 'Channel 2/3 Ratio by Group', 'channel_2_3_ratio')}
            </div>
            
            {data && data.mouse_averages && (
              <div className="summary-stats">
                <h3>Summary</h3>
                <p><strong>Mice:</strong> {data.mouse_averages.length}</p>
                <p><strong>Groups:</strong> {[...new Set(data.mouse_averages.map(d => d.group))].join(', ')}</p>
                <p><strong>Thresholds:</strong> Ch1={thresholds.channel_1}, Ch2={thresholds.channel_2}, Ch3={thresholds.channel_3}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
