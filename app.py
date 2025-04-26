import streamlit as st
import pandas as pd
import joblib
import requests
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Powerplant Impact Analysis", layout="wide", initial_sidebar_state="auto")


st.markdown("""
<style>
.stPlotlyChart > div {
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
    border-radius: 40px;
    overflow: hidden;
    transition: all 0.5s ease-in-out;
    padding-top: 2rem;
    height: 100% !important;
    width: 100% !important;
}
.stPlotlyChart > div:hover {
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.2);
}

.stPlotlyChart .modebar {
    top: 10px !important;
    right: 20px !important;
}

.stPlotlyChart .gtitle, .stPlotlyChart .gtitle-main {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.streamlit-expanderHeader {
    font-weight: bold;
}
</style>

""", unsafe_allow_html=True)

# Load trained pipeline
pipeline = joblib.load("impact_classifier_pipeline_tuned.pkl")

# Extract context from milestone PDFs
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# Parse milestone 1 and 2 PDFs
m1_text = extract_pdf_text("Milestone1.pdf")
m2_text = extract_pdf_text("Milestone2.pdf")

milestone_context = m1_text + "\n\n" + m2_text

# load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Impact API call using REST
def ask_Impact(prompt):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"❌ Impact API Error: {str(e)}"

def predict_impact(input_data):
    df = pd.DataFrame([input_data])
    return pipeline.predict(df)[0]    


def generate_data_insights():
    df_merged = pd.read_csv("final_merged_dataset.csv")
    df_labels = pd.read_csv("Dataset_with_Impact_Labels.csv")

    numeric_cols = df_merged.select_dtypes(include=['float64', 'int64']).columns
    correlations = df_merged[numeric_cols].corr().round(3)

    impact_distribution = df_labels['impact_label'].value_counts().to_dict()

    fuel_metrics = df_merged.groupby('primary_fuel').agg({
        'max_aqi': 'mean',
        'avg_temp': 'mean',
        'capacity_mw': 'mean'
    }).round(2).to_dict()
    
    feature_impact = {}
    if 'delta_aqi' in df_labels.columns and 'impact_label' in df_labels.columns:
        for label in df_labels['impact_label'].unique():
            subset = df_labels[df_labels['impact_label'] == label]
            feature_impact[label] = {
                'avg_delta_aqi': subset['delta_aqi'].mean(),
                'avg_delta_temp': subset['delta_temp'].mean(),
                'count': len(subset)
            }
    
    return {
        'correlations': correlations,
        'impact_distribution': impact_distribution,
        'fuel_metrics': fuel_metrics,
        'feature_impact': feature_impact
    }

def enhanced_ask_Impact(user_query, include_data=True):

    if include_data:
        insights = generate_data_insights()
        data_context = f"""
        DATA INSIGHTS:
        - Impact Label Distribution: {insights['impact_distribution']}
        - Feature Correlations: {dict(insights['correlations'].max())}
        - Metrics by Fuel Type: {insights['fuel_metrics']}
        - Feature-Impact Relationships: {insights['feature_impact']}
        """
    else:
        data_context = ""
    
    full_prompt = f"""You are an expert environmental data analyst specializing in power plant impact assessment.

    ROLE: You provide data-driven insights about how different types of power plants affect environmental metrics like air quality and temperature. You analyze correlations, explain impact classifications, and help users understand the environmental implications of power generation.

    INPUT CONTEXT:
    1. PROJECT RESEARCH: {milestone_context}
    2. STATISTICAL DATA: {data_context}

    RESPONSE GUIDELINES:
    - Present numerical insights if possible with appropriate units (MW, °C, AQI values)
    - For questions about specific fuel types, compare with averages across all plants
    - When discussing environmental impact, explain both the statistical basis and real-world implications
    - Use a professional but accessible tone suitable for energy sector professionals and environmental researchers
    - Format lists and comparisons clearly to enhance readability

    SPECIAL INSTRUCTIONS:
    - For correlation questions: Provide exact correlation values and explain their practical significance
    - For impact predictions: Explain which factors most strongly influence the model's classifications 
    - For fuel comparisons: Present data on capacity, AQI, and temperature differences between fuel types
    - For trend questions: Describe both the statistical trend and potential causal factors

    DO NOT:
    - Reference document sources like "according to the PDF" or "as shown in Figure X"
    - Present opinions on which energy sources are "best" without statistical backing

    Question: {user_query}
.
    If asked about specifics not in the context, you can run basic analysis on the fly using pandas syntax.
    """
    
    return ask_Impact(full_prompt)


# Sidebar Navigation
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to", ["Predictive Tool", "Impact Chatbot", "Data Dashboard"])

if page == "Predictive Tool":
    st.title("Power Plant Environmental Impact Predictor")
    
    renewable_fuels = ["solar", "wind", "hydro", "biomass", "geothermal"]
    non_renewable_fuels = ["coal", "gas", "oil", "waste", "storage", "cogeneration", "other"]
    all_fuels = renewable_fuels + non_renewable_fuels
    
    primary_fuel = st.selectbox(
        "Primary Fuel", 
        all_fuels,
        help="Select the primary fuel for the power plant"
    )
    
    is_renewable = primary_fuel in renewable_fuels
    st.write(f"Selected fuel type is {'✅ renewable' if is_renewable else '⚠️ non-renewable'}")

    with st.form("prediction_form"):
        pre_aqi = st.number_input("Pre-commissioning AQI", 0.0, 500.0, 50.0)
        pre_temp = st.number_input("Pre-commissioning Temperature (°C)", -50.0, 60.0, 25.0)
        plant_age = st.number_input("Plant Age (years)", 0, 100, 10)

        st.markdown(f"**Selected fuel:** {primary_fuel}")
        
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = {
                'pre_aqi': pre_aqi,
                'pre_temp': pre_temp,
                'plant_age': plant_age,
                'is_renewable': 1 if is_renewable else 0,
                'primary_fuel': primary_fuel
            }
            prediction = predict_impact(input_data)
            st.success(f"Predicted Environmental Impact: {prediction}")

            Impact_prompt = f"""
            Based on the following input:
            - AQI: {pre_aqi}, Temperature: {pre_temp}°C, Plant age: {plant_age} years,
            - Fuel: {primary_fuel} ({"renewable" if is_renewable else "non-renewable"})
            The model predicted a {prediction} environmental impact.
            This prediction was made using a trained machine learning model that identifies patterns based on historical data.
            Please explain the likely reasoning without implying the inputs alone are sufficient.
            Do not talk about Missing Context/Features. Support model prediction with data-driven insights.
            """
            explanation = ask_Impact(Impact_prompt)
            st.info("Impact Explanation:")
            st.write(explanation)


elif page == "Impact Chatbot":
    st.title("Impact Chatbot")
    st.write("Use this chatbot to explore insights about the project methodology and data analysis.")

    include_data = st.checkbox("Include data analysis", value=True, 
                             help="When enabled, the chatbot will analyze the dataset to provide insights")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.chat_message("assistant"):
        st.markdown("Hello! Ask me about the power plant environmental impact project. I can provide insights about data correlations, model predictions, and methodology.")

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    user_query = st.chat_input("Ask about the project, data correlations, or model insights...")
    if user_query:
        bot_response = enhanced_ask_Impact(user_query, include_data)
        st.session_state.chat_history.append((user_query, bot_response))
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            st.markdown(bot_response)

# Dashboard Page
elif page == "Data Dashboard":
    st.title("Powerplant Impact Dashboard")

    df_merged = pd.read_csv("final_merged_dataset.csv")
    df_labels = pd.read_csv("Dataset_with_Impact_Labels.csv")

    st.sidebar.markdown("### Dashboard Controls")
    
    years = sorted(df_merged['year'].unique())
    if len(years) > 1:
        year_range = st.sidebar.slider(
            "Select Year Range", 
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years))
        )
        df_filtered = df_merged[(df_merged['year'] >= year_range[0]) & (df_merged['year'] <= year_range[1])]
    else:
        df_filtered = df_merged

    # Filters: Fuel Types
    fuels = df_merged['primary_fuel'].unique().tolist()
    selected_fuels = st.sidebar.multiselect("Filter Fuel Types", fuels, default=fuels, help="Select fuel categories to display")
    df_filtered = df_merged[df_merged['primary_fuel'].isin(selected_fuels)]
    
    # Capacity range filter
    capacity_range = st.sidebar.slider(
        "Capacity Range (MW)",
        min_value=float(df_filtered['capacity_mw'].min()),
        max_value=float(df_filtered['capacity_mw'].max()),
        value=(float(df_filtered['capacity_mw'].min()), float(df_filtered['capacity_mw'].max()))
    )
    df_filtered = df_filtered[(df_filtered['capacity_mw'] >= capacity_range[0]) & (df_filtered['capacity_mw'] <= capacity_range[1])]

    
    with st.container():
        col1, col2, col3= st.columns(3)
        col1.metric("Total Plants", df_filtered['name'].nunique(), 
                   delta=f"{df_filtered['name'].nunique() - df_merged['name'].nunique()}" if df_filtered['name'].nunique() != df_merged['name'].nunique() else None)
        
        col2.metric("Avg Capacity (MW)", round(df_filtered['capacity_mw'].mean(), 2), 
                   delta=f"{round(df_filtered['capacity_mw'].mean() - df_merged['capacity_mw'].mean(), 2)}" if round(df_filtered['capacity_mw'].mean(), 2) != round(df_merged['capacity_mw'].mean(), 2) else None)
        
        col3.metric("Avg Max AQI", round(df_filtered['max_aqi'].mean(), 2), 
                   delta=f"{round(df_filtered['max_aqi'].mean() - df_merged['max_aqi'].mean(), 2)}" if round(df_filtered['max_aqi'].mean(), 2) != round(df_merged['max_aqi'].mean(), 2) else None)
        
    # tabs for insights
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Fuel Insights", "Global Map", "AQI & Temp Trends", 
        "Impact Analysis", "Cluster Overview"
    ])

    with tab1:
        st.subheader("Fuel Type Overview")
        palette = ['#0C2D48','#A0C1B8','#F0E5D8','#D6A77A','#A63D40']
        fuel_count = df_filtered['primary_fuel'].value_counts().reset_index()
        fuel_count.columns = ['fuel', 'count']
        fuel_cap = df_filtered.groupby('primary_fuel')['capacity_mw'].sum().reset_index()

        # top owners
        owner_group = df_filtered.groupby('owner').agg(
            total_capacity=('capacity_mw', 'sum'),
            plant_count=('name', 'count')
        ).reset_index()
        top_owners = owner_group.sort_values('total_capacity', ascending=False).head(10)

        # Plant count by fuel
        fig_fuel_pie = px.pie(
            fuel_count, names='fuel', values='count', title="Plant Count by Fuel Type", hole=0.4,
            color_discrete_sequence=palette
        )
        fig_fuel_pie.update_layout(
            template='plotly_dark', 
            transition={'duration':500}, 
            margin=dict(l=80, r=80, t=80, b=40), 
            title_font_size=20,
            title_x=0.05
        )
        st.plotly_chart(fig_fuel_pie, use_container_width=True)

        # Total capacity by fuel
        fig_fuel_bar = px.bar(
            fuel_cap, x='primary_fuel', y='capacity_mw', color='primary_fuel',
            title="Total Capacity per Fuel Type (MW)", labels={'capacity_mw':'Total Capacity (MW)'},
            color_discrete_sequence=palette
        )
        fig_fuel_bar.update_layout(
            template='plotly_dark', 
            transition={'duration':500}, 
            margin=dict(l=80, r=80, t=80, b=40), 
            title_font_size=20,
            title_x=0.05
        )
        st.plotly_chart(fig_fuel_bar, use_container_width=True)

        # Top Owners by Capacity and Plant Count
        fig_owner = px.bar(
            top_owners, x='owner', y=['total_capacity','plant_count'], barmode='group',
            title="Top Owners: Capacity & Plant Count", color_discrete_sequence=palette
        )
        fig_owner.update_layout(
            xaxis_tickangle=-45, 
            template='plotly_dark', 
            transition={'duration':500}, 
            margin=dict(l=80, r=80, t=80, b=40), 
            title_font_size=20,
            title_x=0.05
        )
        st.plotly_chart(fig_owner, use_container_width=True)

    with tab2:
        st.subheader("Plant Locations")
        fuel_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041', '#229954'
        ]
        fig_map = px.scatter_mapbox(
            df_filtered, lat='latitude', lon='longitude', color='primary_fuel', size='capacity_mw',
            hover_name='name', color_discrete_sequence=fuel_palette,
            height=600
        )
        fig_map.update_layout(
            mapbox_style='carto-positron', mapbox_zoom=3, mapbox_center={'lat':37.0902,'lon':-95.7129},
            template='plotly_dark', transition={'duration':500},
            paper_bgcolor='#f8f9fa', plot_bgcolor='#f8f9fa',
            margin=dict(t=80, b=40, l=80, r=80)
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        st.subheader("AQI & Temperature Trends")
        yearly = df_filtered.groupby('year').agg(
            avg_aqi=('max_aqi','mean'), avg_temp=('avg_temp','mean')
        ).reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=yearly['year'], y=yearly['avg_aqi'], name='Avg Max AQI', marker_color='indianred'
        ))
        fig_trend.add_trace(go.Scatter(
            x=yearly['year'], y=yearly['avg_temp'], name='Avg Temp (°C)', yaxis='y2', marker_color='steelblue'
        ))
        fig_trend.update_layout(
            title="Yearly AQI vs Temperature", xaxis_title='Year', yaxis_title='AQI',
            yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right'),
            template='plotly_dark', transition={'duration':500}, 
            title_font_size=20, title_x=0.05,
            margin=dict(l=80, r=80, t=80, b=40)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        fig_anim = px.scatter(
            df_filtered, x='max_aqi', y='capacity_mw', animation_frame='year', color='primary_fuel',
            size='capacity_mw', hover_name='name',
            range_x=[0, df_filtered['max_aqi'].max()*1.1],
            range_y=[0, df_filtered['capacity_mw'].max()*1.1],
            title="AQI vs Capacity Over Time"
        )
        fig_anim.update_layout(
            template='plotly_dark', transition={'duration':500}, 
            title_font_size=20, title_x=0.05,
            margin=dict(l=80, r=80, t=80, b=40)
        )
        st.plotly_chart(fig_anim, use_container_width=True)    

    with tab4:
        st.subheader("Environmental Impact Analysis")
        
        df_labels_modified = df_labels.copy()
        df_labels_modified.loc[df_labels_modified['impact_label'].isin(['Positive', 'Neutral']), 'impact_label'] = 'Neutral/Positive'
        
        # Impact label distribution
        fig_label = px.histogram(
            df_labels_modified, x='impact_label', color='impact_label',
            title="Environmental Impact Distribution"
        )
        fig_label.update_layout(
            template='plotly_dark', transition={'duration':500}, 
            title_font_size=20, title_x=0.05,
            margin=dict(l=80, r=80, t=80, b=40)
        )
        st.plotly_chart(fig_label, use_container_width=True)

        # Change in AQI by impact label
        fig_delta_aqi = px.box(
            df_labels_modified, x='impact_label', y='delta_aqi', points='all',
            title="Delta AQI by Impact Label"
        )
        fig_delta_aqi.update_layout(
            template='plotly_dark', transition={'duration':500}, 
            title_font_size=20, title_x=0.05,
            margin=dict(l=80, r=80, t=80, b=40)
        )
        st.plotly_chart(fig_delta_aqi, use_container_width=True)

    with tab5:
        st.subheader("Cluster Overview")
        
        cluster_data = df_filtered.copy()
        max_capacity = df_filtered['capacity_mw'].max()
        cluster_data['elevation'] = (cluster_data['capacity_mw'] / max_capacity) * 1000
        unique_clusters = cluster_data['cluster'].unique()
        cluster_colors = {}
        colors = [
            [239, 83, 80],   # Red
            [66, 165, 245],  # Blue
            [102, 187, 106], # Green
            [171, 71, 188],  # Purple
            [255, 167, 38],  # Orange
            [41, 182, 246],  # Light Blue
        ]
        
        for i, cluster in enumerate(unique_clusters):
            cluster_colors[cluster] = colors[i % len(colors)]
 
        cluster_data['color'] = cluster_data['cluster'].map(lambda c: cluster_colors.get(c, [100, 100, 100]))
        
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=cluster_data,
            get_position=["longitude", "latitude"],
            get_elevation='capacity_mw',
            elevation_scale=1000,
            pickable=True,
            extruded=True,
            radius=5000, 
            disk_resolution=12,
            get_fill_color="color",
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            longitude=-95.7129,
            latitude=37.0902,
            zoom=4,
            min_zoom=2,
            max_zoom=15,
            pitch=65, 
            bearing=0,
        )

        r = pdk.Deck(
            layers=[column_layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={
                "html": "<b>Name:</b> {name}<br><b>Cluster:</b> {cluster}<br><b>Capacity:</b> {capacity_mw} MW",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        )
        st.pydeck_chart(r, use_container_width=True)

#-------------------------------------------------------------------------------------------------------
