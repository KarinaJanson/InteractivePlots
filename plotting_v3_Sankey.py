import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime

# Function to upload and read the Excel file
def upload_file():
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Dataframe preview:")
        st.write(df.head())
        return df
    return None

# Function to select columns for the plot
def select_columns(df):
    return st.multiselect("Select columns for the plot", df.columns.tolist())

# Function to select plot type
def select_plot_type():
    return st.selectbox("Select Plot Type", ["Sunburst", "Treemap", "Sankey"])

# Function to validate the column selection
def validate_columns(columns, plot_type):
    if plot_type == "Sankey" and len(columns) < 2:
        st.warning("Please select at least 2 columns for the Sankey plot.")
        return False
    elif plot_type != "Sankey" and len(columns) < 3:
        st.warning("Please select at least 3 columns for Sunburst and Treemap plots.")
        return False
    return True

# Function to select layers for the plot
def select_layers(columns):
    col1 = st.selectbox("Select First Layer", columns, key='col1')
    col2 = st.selectbox("Select Second Layer", columns, key='col2')
    col3 = st.selectbox("Select Third Layer", columns, key='col3')
    col4 = st.selectbox("Select Fourth Layer (optional)", [None] + columns, key='col4')
    return col1, col2, col3, col4

# Function to preprocess the DataFrame for hierarchical plots
def preprocess_dataframe_for_hierarchy(df, columns):
    path_cols = columns
    df_grouped = df.groupby(path_cols).size().reset_index(name='value')
    return df_grouped

# Function to generate and display the plot with a fixed color palette
def generate_plot(df, plot_type, col1, col2, col3, col4):
    color_discrete_sequence = px.colors.qualitative.Plotly  # Fixed color palette
    columns = [col for col in [col1, col2, col3, col4] if col]
    
    if plot_type in ["Sunburst", "Treemap"]:
        df_preprocessed = preprocess_dataframe_for_hierarchy(df, columns)
        if plot_type == "Sunburst":
            fig = px.sunburst(df_preprocessed, path=columns, values='value', color_discrete_sequence=color_discrete_sequence)
        elif plot_type == "Treemap":
            fig = px.treemap(df_preprocessed, path=columns, values='value', color_discrete_sequence=color_discrete_sequence)
    elif plot_type == "Sankey":
        fig = generate_sankey(df, columns)
        
    st.plotly_chart(fig)
    return fig

# Function to generate Sankey plot with dynamic columns
def generate_sankey(df, columns):
    all_labels = pd.concat([df[col] for col in columns]).unique()
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    sources = []
    targets = []
    values = []
    
    for i in range(len(columns) - 1):
        source_col = columns[i]
        target_col = columns[i + 1]
        
        grouped = df.groupby([source_col, target_col]).size().reset_index(name='value')
        
        sources.extend(grouped[source_col].map(label_map))
        targets.extend(grouped[target_col].map(label_map))
        values.extend(grouped['value'])

    link = dict(source=sources, target=targets, value=values)
    node = dict(label=list(label_map.keys()), pad=15, thickness=20)

    fig = go.Figure(data=[go.Sankey(link=link, node=node)])
    fig.update_layout(title_text="Sankey Diagram", font_size=10)
    return fig

# Function to save the plot as an HTML file with a timestamped filename
def save_plot_as_html(fig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{timestamp}.html"
    pio.write_html(fig, file=filename, auto_open=False)
    st.success(f"Plot saved as {filename}")

# Function to reset selections
def reset_selections():
    st.button("Reset Selections", on_click=reset_columns)

# Function to reset column selections (stub for illustration)
def reset_columns():
    st.experimental_rerun()

# Function to style the application
def style_app():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Title of the Streamlit app
st.title("Data Visualization: Sunburst, Treemap, and Sankey Plots")

# Apply custom styles
style_app()

# Upload file and display dataframe
df = upload_file()

if df is not None:
    # Select columns for the plot
    columns = select_columns(df)

    if columns:
        # Select plot type
        plot_type = select_plot_type()

        if validate_columns(columns, plot_type):
            # Select layers for the plot
            st.write("### Select Layers for the Plot")
            col1, col2, col3, col4 = select_layers(columns)

            # Button to generate the plot
            if st.button(f"Generate {plot_type} Plot"):
                fig = generate_plot(df, plot_type, col1, col2, col3, col4)
                # Save the figure to session state
                st.session_state['fig'] = fig

            # Display save plot button only if a plot has been generated
            if 'fig' in st.session_state:
                if st.button("Save Plot as HTML"):
                    save_plot_as_html(st.session_state['fig'])

            # Reset selections
            reset_selections()
