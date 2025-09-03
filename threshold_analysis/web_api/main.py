"""FastAPI application for interactive threshold analysis."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
from scipy import stats
import numpy as np

# MINIMAL: Start with in-memory data for testing
app = FastAPI(title="ND2 Interactive Threshold Analysis", version="0.1.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MINIMAL data models
class ThresholdRequest(BaseModel):
    channel_1: int
    channel_2: int 
    channel_3: int

class MouseAverage(BaseModel):
    group: str
    mouse_id: str
    channel_1_area: float
    channel_2_area: float
    channel_3_area: float

class StatisticalRequest(BaseModel):
    channel_1: int
    channel_2: int
    channel_3: int
    comparison_mode: str  # "all_vs_one" or "pairs"
    reference_group: Optional[str] = None  # For all_vs_one mode
    comparison_pairs: Optional[List[Tuple[str, str]]] = None  # For pairs mode
    test_type: str = "auto"  # "parametric", "non_parametric", or "auto"
    significance_display: str = "stars"  # "stars" or "p_values"

# Global storage for loaded threshold data
LOADED_STUDIES = {}

def perform_normality_test(data: List[float]) -> bool:
    """Test for normality using Shapiro-Wilk test."""
    if len(data) < 3:
        return True  # Assume normal for small samples
    try:
        _, p_value = stats.shapiro(data)
        return p_value > 0.05  # Normal if p > 0.05
    except:
        return True

def perform_statistical_test(group1: List[float], group2: List[float], test_type: str = "auto") -> Tuple[float, float]:
    """Perform appropriate statistical test between two groups."""
    if test_type == "auto":
        # Auto-detect based on normality and sample size
        normal1 = perform_normality_test(group1)
        normal2 = perform_normality_test(group2)
        use_parametric = normal1 and normal2 and len(group1) >= 3 and len(group2) >= 3
    else:
        use_parametric = test_type == "parametric"
    
    if use_parametric:
        # Independent t-test
        statistic, p_value = stats.ttest_ind(group1, group2)
    else:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    return float(statistic), float(p_value)

def p_value_to_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def perform_anova(groups_data: Dict[str, List[float]], test_type: str = "auto") -> Tuple[float, float]:
    """Perform ANOVA or Kruskal-Wallis test."""
    group_values = list(groups_data.values())
    
    if test_type == "auto":
        # Check normality for all groups
        all_normal = all(perform_normality_test(group) for group in group_values)
        use_parametric = all_normal and all(len(group) >= 3 for group in group_values)
    else:
        use_parametric = test_type == "parametric"
    
    if use_parametric:
        # One-way ANOVA
        statistic, p_value = stats.f_oneway(*group_values)
    else:
        # Kruskal-Wallis test (non-parametric ANOVA)
        statistic, p_value = stats.kruskal(*group_values)
    
    return float(statistic), float(p_value)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "ND2 Interactive Threshold Analysis API", "version": "0.1.0"}

class LoadStudyRequest(BaseModel):
    file_path: str

@app.post("/api/studies/load")
async def load_study(request: LoadStudyRequest):
    """Load a study from threshold results file."""
    file_path = request.file_path
    try:
        from threshold_analysis.batch_processor import load_threshold_results
        
        results = load_threshold_results(file_path)
        study_id = results.study_name.replace(' ', '_').lower()
        
        LOADED_STUDIES[study_id] = results
        
        return {
            "study_id": study_id,
            "name": results.study_name,
            "groups": list(results.group_info.keys()),
            "total_images": len(results.image_data),
            "mice_count": len(set(img.mouse_id for img in results.image_data))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading study: {str(e)}")

@app.get("/api/studies")
async def list_studies():
    """List loaded studies."""
    if not LOADED_STUDIES:
        # Return mock data if no studies loaded
        return [{"id": "mock_study", "name": "Mock Study", "groups": ["Control", "Treatment"]}]
    
    studies = []
    for study_id, results in LOADED_STUDIES.items():
        studies.append({
            "id": study_id,
            "name": results.study_name,
            "groups": list(results.group_info.keys()),
            "total_images": len(results.image_data),
            "mice_count": len(set(img.mouse_id for img in results.image_data))
        })
    
    return studies

@app.post("/api/studies/{study_id}/analyze")
async def analyze_thresholds(study_id: str, thresholds: ThresholdRequest):
    """Analyze data with given thresholds."""
    # Check if study is loaded - if not, try to find any loaded study
    if study_id not in LOADED_STUDIES and LOADED_STUDIES:
        # Use the first loaded study if specific ID not found
        study_id = list(LOADED_STUDIES.keys())[0]
    
    if study_id not in LOADED_STUDIES:
        # Return mock data with your 6 groups for testing
        mock_results = [
            MouseAverage(group="Neg", mouse_id="M00", channel_1_area=2.1, channel_2_area=1.5, channel_3_area=15.2),
            MouseAverage(group="HbSS ATIII", mouse_id="Y36", channel_1_area=8.3, channel_2_area=6.1, channel_3_area=28.5),
            MouseAverage(group="HbSS ATIII", mouse_id="Y65", channel_1_area=12.1, channel_2_area=9.3, channel_3_area=32.1),
            MouseAverage(group="HbSS 300Ug-kg", mouse_id="Y34", channel_1_area=15.2, channel_2_area=11.8, channel_3_area=35.7),
            MouseAverage(group="HbSS 300Ug-kg", mouse_id="Y63", channel_1_area=18.5, channel_2_area=14.2, channel_3_area=38.9),
            MouseAverage(group="HbSS 100Ug-kg", mouse_id="Y33", channel_1_area=10.8, channel_2_area=8.7, channel_3_area=29.3),
            MouseAverage(group="HbSS 100Ug-kg", mouse_id="X48", channel_1_area=13.6, channel_2_area=10.9, channel_3_area=33.1),
            MouseAverage(group="HbSS Untreated", mouse_id="Y80", channel_1_area=22.4, channel_2_area=18.7, channel_3_area=42.8),
            MouseAverage(group="HbSS Untreated", mouse_id="Y44", channel_1_area=25.1, channel_2_area=21.3, channel_3_area=45.2),
            MouseAverage(group="HbAA Untreated", mouse_id="Y21", channel_1_area=5.8, channel_2_area=4.2, channel_3_area=19.6),
            MouseAverage(group="HbAA Untreated", mouse_id="Y23", channel_1_area=7.3, channel_2_area=5.8, channel_3_area=22.4)
        ]
        return {"mouse_averages": [r.dict() for r in mock_results]}
    
    try:
        # Get real data from loaded study
        results = LOADED_STUDIES[study_id]
        
        # Convert thresholds to dict format
        threshold_dict = {
            'channel_1': thresholds.channel_1,
            'channel_2': thresholds.channel_2,
            'channel_3': thresholds.channel_3
        }
        
        # Calculate mouse averages using real data
        mouse_averages_df = results.get_mouse_averages(threshold_dict)
        
        # Add ratio calculations
        mouse_averages_df['Channel_1_3_ratio'] = mouse_averages_df['Channel_1_area'] / (mouse_averages_df['Channel_3_area'] + 0.001)
        mouse_averages_df['Channel_2_3_ratio'] = mouse_averages_df['Channel_2_area'] / (mouse_averages_df['Channel_3_area'] + 0.001)
        
        # Convert to API format and include individual image data
        mouse_averages = []
        individual_images = []
        
        for _, row in mouse_averages_df.iterrows():
            mouse_averages.append({
                "group": row['Group'],
                "mouse_id": row['MouseID'],
                "channel_1_area": float(row['Channel_1_area']),
                "channel_2_area": float(row['Channel_2_area']),
                "channel_3_area": float(row['Channel_3_area']),
                "channel_1_3_ratio": float(row['Channel_1_3_ratio']),
                "channel_2_3_ratio": float(row['Channel_2_3_ratio'])
            })
            
            # Get individual images for this mouse
            mouse_images = [img for img in results.image_data if img.mouse_id == row['MouseID']]
            for img in mouse_images:
                ch1_val = img.get_percentage_at_threshold(1, thresholds.channel_1)
                ch2_val = img.get_percentage_at_threshold(2, thresholds.channel_2) 
                ch3_val = img.get_percentage_at_threshold(3, thresholds.channel_3)
                
                individual_images.append({
                    "group": img.group,
                    "mouse_id": img.mouse_id,
                    "filename": img.filename,
                    "channel_1_area": float(ch1_val),
                    "channel_2_area": float(ch2_val),
                    "channel_3_area": float(ch3_val),
                    "channel_1_3_ratio": float(ch1_val / (ch3_val + 0.001)),
                    "channel_2_3_ratio": float(ch2_val / (ch3_val + 0.001))
                })
        
        return {
            "mouse_averages": mouse_averages,
            "individual_images": individual_images
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing thresholds: {str(e)}")

@app.post("/api/studies/{study_id}/statistics")
async def analyze_statistics(study_id: str, request: StatisticalRequest):
    """Perform statistical analysis with given thresholds and comparison settings."""
    # Check if study is loaded - if not, try to find any loaded study
    if study_id not in LOADED_STUDIES and LOADED_STUDIES:
        study_id = list(LOADED_STUDIES.keys())[0]
    
    if study_id not in LOADED_STUDIES:
        raise HTTPException(status_code=404, detail="No study data loaded")
    
    try:
        # Get real data from loaded study
        results = LOADED_STUDIES[study_id]
        
        # Convert thresholds to dict format
        threshold_dict = {
            'channel_1': request.channel_1,
            'channel_2': request.channel_2,
            'channel_3': request.channel_3
        }
        
        # Calculate mouse averages using real data
        mouse_averages_df = results.get_mouse_averages(threshold_dict)
        
        # Organize data by groups for each channel and calculate ratios
        channels = ['channel_1', 'channel_2', 'channel_3', 'channel_1_3_ratio', 'channel_2_3_ratio']
        statistics_results = {}
        
        # Add ratio calculations
        mouse_averages_df['Channel_1_3_ratio'] = mouse_averages_df['Channel_1_area'] / (mouse_averages_df['Channel_3_area'] + 0.001)  # Add small value to avoid division by zero
        mouse_averages_df['Channel_2_3_ratio'] = mouse_averages_df['Channel_2_area'] / (mouse_averages_df['Channel_3_area'] + 0.001)
        
        for channel in channels:
            if channel.endswith('_ratio'):
                # For ratio channels, use the exact column name from DataFrame
                if channel == 'channel_1_3_ratio':
                    channel_col = 'Channel_1_3_ratio'
                elif channel == 'channel_2_3_ratio':
                    channel_col = 'Channel_2_3_ratio'
            else:
                channel_col = f'Channel_{channel.split("_")[1]}_area'
            groups_data = {}
            
            # Group data by treatment group
            for group in mouse_averages_df['Group'].unique():
                group_data = mouse_averages_df[mouse_averages_df['Group'] == group][channel_col].tolist()
                groups_data[group] = group_data
            
            # Perform statistical analysis based on comparison mode
            if request.comparison_mode == "all_vs_one":
                if not request.reference_group or request.reference_group not in groups_data:
                    # Use first group as reference if not specified
                    reference_group = list(groups_data.keys())[0]
                else:
                    reference_group = request.reference_group
                
                comparisons = []
                reference_data = groups_data[reference_group]
                
                for group_name, group_data in groups_data.items():
                    if group_name != reference_group:
                        statistic, p_value = perform_statistical_test(
                            reference_data, group_data, request.test_type
                        )
                        
                        significance = p_value_to_stars(p_value) if request.significance_display == "stars" else f"p={p_value:.4f}"
                        
                        comparisons.append({
                            "comparison": f"{reference_group} vs {group_name}",
                            "statistic": statistic,
                            "p_value": p_value,
                            "significance": significance,
                            "group1": reference_group,
                            "group2": group_name
                        })
                
                # Overall ANOVA
                anova_stat, anova_p = perform_anova(groups_data, request.test_type)
                anova_sig = p_value_to_stars(anova_p) if request.significance_display == "stars" else f"p={anova_p:.4f}"
                
                statistics_results[channel] = {
                    "comparison_mode": "all_vs_one",
                    "reference_group": reference_group,
                    "overall_test": {
                        "statistic": anova_stat,
                        "p_value": anova_p,
                        "significance": anova_sig
                    },
                    "pairwise_comparisons": comparisons
                }
                
            elif request.comparison_mode == "pairs":
                if not request.comparison_pairs:
                    raise HTTPException(status_code=400, detail="Comparison pairs must be specified for pairs mode")
                
                comparisons = []
                for group1, group2 in request.comparison_pairs:
                    if group1 in groups_data and group2 in groups_data:
                        statistic, p_value = perform_statistical_test(
                            groups_data[group1], groups_data[group2], request.test_type
                        )
                        
                        significance = p_value_to_stars(p_value) if request.significance_display == "stars" else f"p={p_value:.4f}"
                        
                        comparisons.append({
                            "comparison": f"{group1} vs {group2}",
                            "statistic": statistic,
                            "p_value": p_value,
                            "significance": significance,
                            "group1": group1,
                            "group2": group2
                        })
                
                statistics_results[channel] = {
                    "comparison_mode": "pairs",
                    "pairwise_comparisons": comparisons
                }
        
        return {
            "statistics": statistics_results,
            "test_type_used": request.test_type,
            "significance_display": request.significance_display,
            "thresholds": threshold_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing statistical analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
