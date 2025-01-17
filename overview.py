import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
st.image("Header.png", use_column_width= True)
st.subheader("Mental Health Crisis Overview")
sdf= pd.read_excel("mental_health_illness_usa.xlsx")
sdf= sdf.sort_values(by= 'Percentage', ascending= False)
sdf= sdf.reset_index()
sdf['Color'] = np.where(
    sdf.index < 5,  # Top 5 rows (highest values)
    "red",
    np.where(sdf['Percentage'] > sdf['Percentage'].mean(), "darkgreen", "lightgreen")
)
fig1= px.choropleth(
    sdf,
    locations="Code",
    locationmode="USA-states",
    color="Color",
    scope="usa",
    color_discrete_map={
        "red": "red",
        "darkgreen": "#00cc00",  # Dark green
        "lightgreen": "#b2fba5"  # Light green
    },
    hover_name= 'State',
    hover_data= 'Percentage',
    custom_data= ['Percentage']
)
fig1.update_traces(
    hovertemplate=
    '<b>%{location}</b><br>'  # State name
    + 'Percentage: %{customdata[0]:.2f}%<br>'  # Percentage
    + '<extra></extra>'  # Hide the color bar info
)
fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',}, title=dict(text="State Population Mental Illness Percentage in 2024", font=dict(size=24)), showlegend= False)
st.plotly_chart(fig1)
st.write("As of 2024, at least 16% of every state's population is suffering from a mental illness, up to a maximum of almost 27%. The states marked in red (Washington, Ohio, Oregon, West Virginia) suffer the most from these ailments.")
# Data
disorders = ["Anxiety Disorders", "Major Depression", "PTSD", "Bipolar Disorder", "BPD", "Eating Disorders", "OCD"]
proportions = [0.191, 0.083, 0.036, 0.028, 0.014, 0.012, 0.012]

# Colors: First two red, the rest light gray
colors = ['red', 'red'] + ['lightgray'] * (len(disorders) - 2)

# Create the figure
fig = go.Figure()

fig.add_trace(go.Bar(
    x=disorders,
    y=proportions,
    marker_color=colors,
    text=[f"{p*100:.1f}%" for p in proportions],  # Add percentages above bars
    textposition="outside"  # Display text above the bars
))

# Update layout
fig.update_layout(
    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',},
    title=dict(text="Mental Illness in the U.S in 2024", font=dict(size=24)),
    xaxis=dict(title="Disorder", title_font=dict(size=18)),
    yaxis=dict(title="Proportion", title_font=dict(size=18)),
    yaxis_range=[0,0.23],
    template="plotly_white"
)

# Display in Streamlit
st.plotly_chart(fig)
st.write("As of 2024, the most common mental illnesses were anxiety disorders and major depression which accounted for 27.4% of people suffering from a mental health issue.")
apdf= pd.read_csv("data-348Bd.csv") 
years= apdf['Demographic Characteristic']
ag1825= apdf['Ages 18-25']
ag2649= apdf['Ages 26-49']
ag50o= apdf['Ages 50 or Older']
agtot= apdf['Total']
fig2 = go.Figure()

# Add traces for each line
fig2.add_trace(go.Scatter(x=years, y=ag1825, mode='lines+markers', name='Age 18-25',
                         line=dict(color='#30bdcc', width=3),
                         marker=dict(size=8)))
fig2.add_trace(go.Scatter(x=years, y=ag2649, mode='lines', name='Age 26-49',
                         line=dict(color='#364059', width=1)))
fig2.add_trace(go.Scatter(x=years, y=ag50o, mode='lines', name='Age 50+',
                         line=dict(color='#2c7eb4', width=1)))
fig2.add_trace(go.Scatter(x=years, y=agtot, mode='lines', name='Overall Population',
                         line=dict(color='#a9545e', width=1)))

# Highlight the most emphasized line
for i, year in enumerate(years):
    fig2.add_annotation(
        x=year, y=ag1825[i], text=f"{ag1825[i]}%", showarrow=False,
        font=dict(size=10 if i < len(years) - 1 else 18, color='black' if i < len(years) - 1 else 'red'),
        yshift=10
    )

# Update layout to emphasize the main plot features
fig2.update_layout(
{'plot_bgcolor': 'rgba(255, 255, 255, 1)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',},
title=dict(text='Mental Illness Percentage By Age Group', font=dict(size=24)),
xaxis=dict(title='Year', tickmode='linear', dtick=1),
yaxis=dict(title='Total Percentage'),
legend=dict(title='Age Group', font=dict(size=14)),
template='plotly_white'
)
st.plotly_chart(fig2)
st.write("The youth of the United States suffers disproportionately from mental illnesses in comparison to the other age groups, with the percentage almost doubling from 2008-2021. It can be inferred that this percentage has only gone up since then, with around 2 in 5 adolescents/young adults having mental health issues.")
MHC= {'Year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], 
      'Funding (Billions USD)': [2.174, 2.213, 2.263, 2.454, 2.717, 3.010, 3.296, 3.577, 3.666, 4.008, 4.162, 4.230, 4.458]}
mhcdf= pd.DataFrame(MHC)
years= mhcdf['Year']
funds= mhcdf['Funding (Billions USD)']
fig3= go.Figure()
fig3.add_trace(go.Scatter(x=years, y=funds, mode='lines+markers', name='Funding',
                         line=dict(color='#2c7eb4', width=3),
                         marker=dict(size=8)))
for i, year in enumerate(years):
    fig3.add_annotation(
        x=year, y=funds[i], text=f"{funds[i]}%", showarrow=False,
        font=dict(size=10 if i < len(years) - 1 else 18, color='black' if i < len(years) - 1 else 'red'),
        yshift=10
    )
fig3.update_layout(
{'plot_bgcolor': 'rgba(255, 255, 255, 1)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',},
title=dict(text='Mental Health Spendings from 2013-2025', font=dict(size=24)),
xaxis=dict(title='Year', tickmode='linear', dtick=1),
yaxis=dict(title='Spending (Billions USD)'),
yaxis_range=[2,5],
template='plotly_white'
)
st.plotly_chart(fig3)
st.write("The annual mental health spending by the National Institue of Health has more than doubled since 2013, with 2025's projected spendings expected to be around almost $4.5 billion USD.")