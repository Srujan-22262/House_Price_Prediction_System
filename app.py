import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model_and_features():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, list(feature_names)

model, feature_names = load_model_and_features()

def predict_price(total_sqft, bath, balcony, bhk, location, area_type):
    x = np.zeros(len(feature_names))
    x[feature_names.index("total_sqft")] = total_sqft
    x[feature_names.index("bath")] = bath
    x[feature_names.index("balcony")] = balcony
    x[feature_names.index("bhk")] = bhk
    loc_col = f"location_{location}"
    if loc_col in feature_names:
        x[feature_names.index(loc_col)] = 1
    area_col = f"area_type_{area_type}"
    if area_col in feature_names:
        x[feature_names.index(area_col)] = 1
    return model.predict([x])[0]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: #667eea !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #764ba2 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Sidebar Labels */
    [data-testid="stSidebar"] label {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: #e0e0e0 !important;
    }
    
    /* Number Input Styling */
    .stNumberInput input {
        background: #f8f9fa !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        color: #333333 !important;
        font-weight: 600 !important;
    }
    
    .stNumberInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        background: #f8f9fa !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: #f8f9fa !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #333333 !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.9rem 1.5rem !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #333333 !important;
    }
    
    [data-testid="metric-container"] {
        background: white !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Horizontal Rule */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* Dataframe Styling */
    [data-testid="stDataFrame"] {
        background: white !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0; background:linear-gradient(135deg,#667eea,#764ba2); 
    border-radius:15px; margin-bottom:1.5rem;'>
    <h1 style='color:white; font-size:1.8rem; margin:0; font-weight:800;'>🏠 Property Config</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background:#f0f2f6; padding:1rem; border-radius:10px; margin-bottom:1rem;'>
    <p style='margin:0; color:#667eea; font-weight:600; font-size:0.9rem;'>
    📐 PROPERTY SIZE</p>
    </div>
    """, unsafe_allow_html=True)
    total_sqft = st.slider("Total Square Feet", 200, 10000, 1000, 50)
    
    st.markdown("""
    <div style='background:#f0f2f6; padding:1rem; border-radius:10px; margin:1.5rem 0 1rem 0;'>
    <p style='margin:0; color:#667eea; font-weight:600; font-size:0.9rem;'>
    🛏 ROOMS & AMENITIES</p>
    </div>
    """, unsafe_allow_html=True)
    bhk = st.selectbox("🏠 BHK (Bedrooms)", [1,2,3,4,5,6,7,8,9,10], index=1)
    
    col1, col2 = st.columns(2)
    with col1:
        bath = st.number_input("🛁 Bathrooms", 1, 10, 2)
    with col2:
        balcony = st.number_input("🌅 Balconies", 0, 5, 1)
    
    st.markdown("""
    <div style='background:#f0f2f6; padding:1rem; border-radius:10px; margin:1.5rem 0 1rem 0;'>
    <p style='margin:0; color:#667eea; font-weight:600; font-size:0.9rem;'>
    📍 LOCATION DETAILS</p>
    </div>
    """, unsafe_allow_html=True)
    locations = sorted([col.replace("location_", "") for col in feature_names if col.startswith("location_")])
    location = st.selectbox("🗺 Select Location", locations)
    
    st.markdown("""
    <div style='background:#f0f2f6; padding:1rem; border-radius:10px; margin:1.5rem 0 1rem 0;'>
    <p style='margin:0; color:#667eea; font-weight:600; font-size:0.9rem;'>
    🏢 PROPERTY TYPE</p>
    </div>
    """, unsafe_allow_html=True)
    area_types = sorted([col.replace("area_type_", "") for col in feature_names if col.startswith("area_type_")])
    area_type = st.selectbox("🏘 Area Type", area_types)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 PREDICT PRICE NOW")
    
    st.markdown("""
    <div style='background:linear-gradient(135deg,#f093fb,#f5576c); padding:1.5rem; 
    border-radius:12px; margin-top:2rem; text-align:center;'>
    <p style='color:white; margin:0; font-size:0.85rem; font-weight:600;'>
    💡 TIP: Adjust parameters to see<br>how they affect the price!</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding:3rem 2rem 2rem 2rem; background:rgba(255,255,255,0.98); border-radius:25px; 
margin-bottom:2rem; box-shadow:0 15px 50px rgba(0,0,0,0.25); position:relative; overflow:visible;'>
<div style='position:absolute; top:-50px; right:-50px; width:200px; height:200px; 
background:linear-gradient(135deg,#667eea,#764ba2); border-radius:50%; opacity:0.1; z-index:0;'></div>
<div style='position:absolute; bottom:-30px; left:-30px; width:150px; height:150px; 
background:linear-gradient(135deg,#f093fb,#f5576c); border-radius:50%; opacity:0.1; z-index:0;'></div>

<h1 style='color:#667eea; font-size:3.5rem; margin-bottom:0.8rem; font-weight:900; position:relative; z-index:1;'>
🏠 Bengaluru House Price Predictor</h1>
<p style='color:#666; font-size:1.3rem; margin:0; font-weight:500; position:relative; z-index:1;'>
✨ AI-Powered Real Estate Price Estimation ✨</p>
<div style='margin-top:1.5rem; display:inline-block; background:linear-gradient(135deg,#667eea,#764ba2); 
padding:0.5rem 2rem; border-radius:20px; position:relative; z-index:1;'>
<p style='color:white; margin:0; font-size:0.9rem; font-weight:600;'>
🎯 Accurate • 🚀 Fast • 💯 Reliable</p>
</div>
</div>

<div style='background:linear-gradient(135deg,#667eea,#764ba2); padding:1.5rem 2rem; border-radius:20px; 
margin-bottom:2rem; box-shadow:0 10px 30px rgba(102,126,234,0.3); text-align:center;'>
<p style='margin:0; color:rgba(255,255,255,0.9); font-size:0.85rem; font-weight:600; 
text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem;'>👥 Developed By</p>
<p style='margin:0; color:white; font-size:1.1rem; font-weight:700; line-height:1.8;'>
Utpal Raj • V Srujan • Surya</p>
<p style='margin:1rem 0 0 0; color:rgba(255,255,255,0.9); font-size:0.85rem; font-weight:600;'>
🎓 Under guidance of <span style='color:white; font-weight:800;'>Sajitha Krishnan</span></p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📋 Property Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📐 Area", f"{total_sqft:,} sq.ft")
with col2:
    st.metric("🛏 Bedrooms", f"{bhk} BHK")
with col3:
    st.metric("🛁 Bathrooms", f"{bath}")
with col4:
    st.metric("🌅 Balconies", f"{balcony}")

st.markdown("---")

if predict_button:
    with st.spinner('🔄 Analyzing property data...'):
        time.sleep(1.5)
        price = predict_price(total_sqft, bath, balcony, bhk, location, area_type)
        price_per_sqft = (price * 100000) / total_sqft
        
        st.markdown(f"""
        <div style='text-align:center; padding:3rem; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); 
        border-radius:20px; margin:2rem 0; box-shadow:0 15px 40px rgba(102,126,234,0.4);'>
        <p style='color:rgba(255,255,255,0.9); font-size:1.2rem; margin-bottom:1rem; text-transform:uppercase; letter-spacing:2px;'>
        Estimated Property Value</p>
        <h1 style='color:white; font-size:4.5rem; margin:0; font-weight:800;'>₹ {price:.2f} L</h1>
        <p style='color:rgba(255,255,255,0.8); font-size:1.1rem; margin-top:1rem;'>
        {location} • {area_type} • {bhk} BHK</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Detailed Price Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style='background:white; padding:1.5rem; border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color:#667eea; margin-top:0;'>💰 Price per Sq.Ft</h4>
            <p style='font-size:2rem; font-weight:700; color:#333; margin:0;'>₹ {price_per_sqft:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background:white; padding:1.5rem; border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color:#667eea; margin-top:0;'>📊 Total Value</h4>
            <p style='font-size:2rem; font-weight:700; color:#333; margin:0;'>₹ {price/100:.2f} Cr</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background:white; padding:1.5rem; border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color:#667eea; margin-top:0;'>🏘 Location</h4>
            <p style='font-size:1.3rem; font-weight:600; color:#333; margin:0;'>{location}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📈 Price Comparison by BHK")
        
        # Calculate prices for different BHK with proportionally scaled areas
        bhk_options = [1,2,3,4,5]
        prices_by_bhk = []
        adjusted_areas = []
        
        # Base area per BHK (typical: 600 sq.ft per bedroom)
        base_area_per_bhk = 600
        
        for b in bhk_options:
            # Calculate realistic area for each BHK
            # 1 BHK: ~600 sqft, 2 BHK: ~1200 sqft, 3 BHK: ~1800 sqft, etc.
            realistic_area = base_area_per_bhk * b
            adjusted_areas.append(realistic_area)
            
            # Predict with realistic area and matching bathroom count
            bathrooms_for_bhk = min(b, bath)  # At least 1 bathroom per BHK
            temp_price = predict_price(realistic_area, bathrooms_for_bhk, balcony, b, location, area_type)
            prices_by_bhk.append(temp_price)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{b} BHK" for b in bhk_options], y=prices_by_bhk,
            marker_color=['#667eea' if b == bhk else '#b8c5f2' for b in bhk_options],
            text=[f"₹{p:.2f}L<br>{int(adjusted_areas[i])} sq.ft" for i, p in enumerate(prices_by_bhk)], 
            textposition='outside',
            textfont=dict(size=12, color='#333333', family='Poppins'),
            hovertemplate='<b>%{x}</b><br>Price: ₹%{y:.2f}L<br>Area: %{text}<extra></extra>'))
        fig.update_layout(
            title=dict(text="Price Comparison by BHK (Standard Area: 600 sq.ft per bedroom)", 
                      font=dict(size=18, color='#333333', family='Poppins')),
            xaxis_title=dict(text="Property Type", font=dict(size=14, color='#333333', family='Poppins')),
            yaxis_title=dict(text="Price (Lakhs)", font=dict(size=14, color='#333333', family='Poppins')),
            xaxis=dict(tickfont=dict(size=12, color='#333333', family='Poppins')),
            yaxis=dict(tickfont=dict(size=12, color='#333333', family='Poppins')),
            plot_bgcolor='rgba(248,249,250,0.5)', 
            paper_bgcolor='rgba(255,255,255,0.98)',
            font=dict(family="Poppins", size=12, color='#333333'), 
            height=450, 
            showlegend=False,
            margin=dict(t=80, b=60, l=60, r=40))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 *Note:* Comparison uses standard area allocation of 600 sq.ft per bedroom. This shows how prices increase with more bedrooms when area is proportional.")
        
        st.markdown("### 📏 Impact of Property Size on Price")
        area_range = np.linspace(500, 3000, 10)
        prices_by_area = [predict_price(a, bath, balcony, bhk, location, area_type) for a in area_range]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=area_range, y=prices_by_area, 
            mode='lines+markers',
            line=dict(color='#667eea', width=4), 
            marker=dict(size=10, color='#764ba2', line=dict(color='white', width=2)),
            fill='tozeroy', 
            fillcolor='rgba(102,126,234,0.3)',
            text=[f"₹{p:.2f}L" for p in prices_by_area],
            hovertemplate='<b>Area:</b> %{x:.0f} sq.ft<br><b>Price:</b> ₹%{y:.2f}L<extra></extra>'))
        fig2.update_layout(
            title=dict(text="Price Trend Based on Property Size", 
                      font=dict(size=18, color='#333333', family='Poppins')),
            xaxis_title=dict(text="Area (sq.ft)", font=dict(size=14, color='#333333', family='Poppins')),
            yaxis_title=dict(text="Price (Lakhs)", font=dict(size=14, color='#333333', family='Poppins')),
            xaxis=dict(tickfont=dict(size=12, color='#333333', family='Poppins'),
                      gridcolor='rgba(200,200,200,0.3)'),
            yaxis=dict(tickfont=dict(size=12, color='#333333', family='Poppins'),
                      gridcolor='rgba(200,200,200,0.3)'),
            plot_bgcolor='rgba(248,249,250,0.5)', 
            paper_bgcolor='rgba(255,255,255,0.98)',
            font=dict(family="Poppins", size=12, color='#333333'), 
            height=450,
            margin=dict(t=80, b=60, l=60, r=40),
            hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### 📝 Property Summary")
        summary_data = {"Feature": ["Location", "Area Type", "Total Area", "Bedrooms", "Bathrooms", 
            "Balconies", "Price per Sq.Ft", "Estimated Price"],
            "Details": [location, area_type, f"{total_sqft:,} sq.ft", f"{bhk} BHK", str(bath), 
            str(balcony), f"₹ {price_per_sqft:,.0f}", f"₹ {price:.2f} Lakhs"]}
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.95); padding:3rem 2rem; border-radius:20px; 
        box-shadow:0 10px 30px rgba(0,0,0,0.1); height:100%;'>
        <h2 style='color:#667eea; font-size:2.2rem; margin-bottom:1.5rem; font-weight:800;'>
        👈 Get Started</h2>
        <p style='color:#666; font-size:1.1rem; line-height:1.8; margin-bottom:2rem;'>
        Configure your property details in the sidebar and click 
        <strong style='color:#667eea;'>"PREDICT PRICE NOW"</strong> to get instant AI-powered estimates!</p>
        <div style='background:linear-gradient(135deg,#667eea,#764ba2); padding:1.5rem; border-radius:12px; margin-top:2rem;'>
        <p style='color:white; margin:0; font-size:1rem; font-weight:600; text-align:center;'>
        🎯 Powered by Advanced Machine Learning</p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.95); padding:3rem 2rem; border-radius:20px; 
        box-shadow:0 10px 30px rgba(0,0,0,0.1); height:100%;'>
        <h3 style='color:#764ba2; margin-top:0; font-size:1.8rem; font-weight:700;'>✨ Key Features</h3>
        <div style='margin:1.5rem 0;'>
        <div style='display:flex; align-items:center; margin:1rem 0;'>
        <div style='background:#667eea; width:40px; height:40px; border-radius:10px; display:flex; 
        align-items:center; justify-content:center; margin-right:1rem;'>
        <span style='font-size:1.5rem;'>⚡</span>
        </div>
        <div>
        <p style='margin:0; color:#333; font-weight:600;'>Real-time Predictions</p>
        <p style='margin:0; color:#999; font-size:0.9rem;'>Instant price estimates</p>
        </div>
        </div>
        <div style='display:flex; align-items:center; margin:1rem 0;'>
        <div style='background:#764ba2; width:40px; height:40px; border-radius:10px; display:flex; 
        align-items:center; justify-content:center; margin-right:1rem;'>
        <span style='font-size:1.5rem;'>📊</span>
        </div>
        <div>
        <p style='margin:0; color:#333; font-weight:600;'>Interactive Charts</p>
        <p style='margin:0; color:#999; font-size:0.9rem;'>Visual price comparisons</p>
        </div>
        </div>
        <div style='display:flex; align-items:center; margin:1rem 0;'>
        <div style='background:#f093fb; width:40px; height:40px; border-radius:10px; display:flex; 
        align-items:center; justify-content:center; margin-right:1rem;'>
        <span style='font-size:1.5rem;'>💡</span>
        </div>
        <div>
        <p style='margin:0; color:#333; font-weight:600;'>Detailed Insights</p>
        <p style='margin:0; color:#999; font-size:0.9rem;'>Comprehensive analysis</p>
        </div>
        </div>
        <div style='display:flex; align-items:center; margin:1rem 0;'>
        <div style='background:#f5576c; width:40px; height:40px; border-radius:10px; display:flex; 
        align-items:center; justify-content:center; margin-right:1rem;'>
        <span style='font-size:1.5rem;'>📈</span>
        </div>
        <div>
        <p style='margin:0; color:#333; font-weight:600;'>Area Impact Analysis</p>
        <p style='margin:0; color:#999; font-size:0.9rem;'>Size vs price trends</p>
        </div>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:white; padding:1rem;'>
<p style='margin:0; font-size:0.9rem;'>🏠 Bengaluru House Price Predictor | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)