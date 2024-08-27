import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Adding TransOrg logo to the sidebar
st.set_page_config(page_title="Trade Promotion Optimization App")
st.sidebar.image("https://transorg.com/wp-content/uploads/2022/04/transorg-logo.png")

# Adding radio buttons for user selection
option = st.sidebar.radio(
    "Choose a Category:",
    ("Tea", "Poha", "Coffee", "Spices", "Masala")
)

#requirement

st.sidebar.write('Business Constraints')
# Sidebar inputs for user-defined parameters
max_investment = st.sidebar.number_input("Maximum Investment", min_value=0, value=100000)
lower_discount = st.sidebar.number_input("Lower Discount Range (%)", min_value=0, value=0)
upper_discount = st.sidebar.number_input("Upper Discount Range (%)", min_value=0, value=10)

# Your data dictionary and DataFrame creation code remains the same
#data1
data1 = {
    'PL4 Sub-Category': [    'KD Red Premium', 
    'Agnileaf Plus', 
    'KD Red Elite', 
    'Tata Gold Classic', 
    'Chakra Gold Care', 
    'Gemini Dust', 
    'Chakra Dust Max', 
    'OI Green Classic', 
    'Chakra Dust Pro', 
    'Leo Blue Edition', 
    'TT Care Tulsi', 
    'Agnileaf Pure', 
    'TT Care Digest Plus',
    'Zenith Green'],
    'Forecast Avg. Base Volume': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Units': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Per Unit COGS': [50, 60, 170, 180, 200, 200, 170, 90, 70, 150, 225, 150, 280, 290],
    'List/ Base Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Per Unit Selling Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Discount': [0.0]*14,
    'Revenue': [72000, 103620, 110000, 198400, 66990, 286280, 195800, 89250, 1700, 40200, 333000, 6000, 137700, 156500],
    'Per Unit Margin': [70, 160, 80, 140, 130, 140, 30, 120, 30, 150, 225, 100, 170, 210],
    'Margin': [42000, 75360, 35200, 86800, 26390, 117880, 29370, 51000, 510, 20100, 166500, 2400, 52020, 65730],
    'Per Unit Rebate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dollor Investment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lower Discount': [lower_discount]*14,
    'Upper Discount': [upper_discount]*14,
    'Discount Uplift': [2.174962489, 1.174962489, 3.00034343, 1.174962489, 1.099384733, 2.174962489, 2.174962489, 2.974962489, 2.56904535, 1.625284225, 1.889298444, 3.038315124, 2.700573718, 5.485022623],
    'Tactic Uplift': [1]*14
}

data2 = {
    'PL4 Sub-Category': [    'Classic Poha Original', 
    'Thick Poha Gold', 
    'Thin Poha Select', 
    'Classic Poha Supreme', 
    'Red Rice Poha Classic', 
    'Classic Poha Superior', 
    'Red Rice Poha Gold', 
    'Fine Poha Standard', 
    'Fine Poha Premium', 
    'Classic Poha Ultra', 
    'Classic Poha Max', 
    'Red Rice Poha Superior', 
    'Classic Poha Elite', 
    'Red Rice Poha Ultra'],
    'Forecast Avg. Base Volume': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Units': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Per Unit COGS': [50, 60, 170, 180, 200, 200, 170, 90, 70, 150, 225, 150, 280, 290],
    'List/ Base Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Per Unit Selling Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Discount': [0.0]*14,
    'Revenue': [72000, 103620, 110000, 198400, 66990, 286280, 195800, 89250, 1700, 40200, 333000, 6000, 137700, 156500],
    'Per Unit Margin': [70, 160, 80, 140, 130, 140, 30, 120, 30, 150, 225, 100, 170, 210],
    'Margin': [42000, 75360, 35200, 86800, 26390, 117880, 29370, 51000, 510, 20100, 166500, 2400, 52020, 65730],
    'Per Unit Rebate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dollor Investment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lower Discount': [lower_discount]*14,
    'Upper Discount': [upper_discount]*14,
    'Discount Uplift': [2.174962489, 1.174962489, 3.00034343, 1.174962489, 1.099384733, 2.174962489, 2.174962489, 2.974962489, 2.56904535, 1.625284225, 1.889298444, 3.038315124, 2.700573718, 5.485022623],
    'Tactic Uplift': [1]*14
}

data3 = {
    'PL4 Sub-Category': [    'Coffee Grand Filter', 
    'Coffee Grand Premium Pack', 
    'Coffee Grand Jar', 
    'Cold Coffee Classic', 
    'Coffee Grand Classic Pack', 
    'Coffee Grand Classic Jar', 
    'Poha Rich', 
    'Coffee Quick Filter', 
    'Coffee Grand Jar Premium', 
    'Grand Coffee Pouch', 
    'Coffee Grand Poly Pack', 
    'Coffee Grand Pack',
    'Classic Coffee Pack',
    'Hot Set Coffee'],
    'Forecast Avg. Base Volume': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Units': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Per Unit COGS': [50, 60, 170, 180, 200, 200, 170, 90, 70, 150, 225, 150, 280, 290],
    'List/ Base Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Per Unit Selling Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Discount': [0.0]*14,
    'Revenue': [72000, 103620, 110000, 198400, 66990, 286280, 195800, 89250, 1700, 40200, 333000, 6000, 137700, 156500],
    'Per Unit Margin': [70, 160, 80, 140, 130, 140, 30, 120, 30, 150, 225, 100, 170, 210],
    'Margin': [42000, 75360, 35200, 86800, 26390, 117880, 29370, 51000, 510, 20100, 166500, 2400, 52020, 65730],
    'Per Unit Rebate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dollor Investment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lower Discount': [lower_discount]*14,
    'Upper Discount': [upper_discount]*14,
    'Discount Uplift': [2.174962489, 1.174962489, 3.00034343, 1.174962489, 1.099384733, 2.174962489, 2.174962489, 2.974962489, 2.56904535, 1.625284225, 1.889298444, 3.038315124, 2.700573718, 5.485022623],
    'Tactic Uplift': [1]*14
}

data4 = {
    'PL4 Sub-Category': [    'Coriander Powder', 
    'Chicken Curry Masala', 
    'Turmeric Powder', 
    'Hanger Garam Masala', 
    'Chickpea Masala', 
    'Premium Chilli Powder', 
    'West Garam Masala', 
    'West Chicken Masala', 
    'Hanger Sambar Masala', 
    'Premium Turmeric Powder', 
    'Chilli Powder', 
    'Sambar Masala', 
    'Coriander Spice',
    'Masala Garam Heavy'],
    'Forecast Avg. Base Volume': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Units': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Per Unit COGS': [50, 60, 170, 180, 200, 200, 170, 90, 70, 150, 225, 150, 280, 290],
    'List/ Base Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Per Unit Selling Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Discount': [0.0]*14,
    'Revenue': [72000, 103620, 110000, 198400, 66990, 286280, 195800, 89250, 1700, 40200, 333000, 6000, 137700, 156500],
    'Per Unit Margin': [70, 160, 80, 140, 130, 140, 30, 120, 30, 150, 225, 100, 170, 210],
    'Margin': [42000, 75360, 35200, 86800, 26390, 117880, 29370, 51000, 510, 20100, 166500, 2400, 52020, 65730],
    'Per Unit Rebate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dollor Investment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lower Discount': [lower_discount]*14,
    'Upper Discount': [upper_discount]*14,
    'Discount Uplift': [2.174962489, 1.174962489, 3.00034343, 1.174962489, 1.099384733, 2.174962489, 2.174962489, 2.974962489, 2.56904535, 1.625284225, 1.889298444, 3.038315124, 2.700573718, 5.485022623],
    'Tactic Uplift': [1]*14
}

data5 = {
    'PL4 Sub-Category': [    'Peri Peri Masala', 
    'Schezwan Fried Rice Masala', 
    'Schezwan Masala', 
    'Chili Chicken Masala', 
    'Pasta Masala', 
    'Chicken 65 Masala', 
    'Hakka Noodle Masala', 
    'Mutter Paneer Masala Mix', 
    'Shahi Paneer Masala Mix', 
    'Chinese Masala Combo', 
    'Chowmein Masala', 
    'Veg Manchurian Masala',
    'chilli Masala',
    'Kashmiri nawab'],
    'Forecast Avg. Base Volume': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Units': [600, 471, 440, 620, 203, 842, 979, 425, 17, 134, 740, 24, 306, 313],
    'Per Unit COGS': [50, 60, 170, 180, 200, 200, 170, 90, 70, 150, 225, 150, 280, 290],
    'List/ Base Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Per Unit Selling Price': [120, 220, 250, 320, 330, 340, 200, 210, 100, 300, 450, 250, 450, 500],
    'Discount': [0.0]*14,
    'Revenue': [72000, 103620, 110000, 198400, 66990, 286280, 195800, 89250, 1700, 40200, 333000, 6000, 137700, 156500],
    'Per Unit Margin': [70, 160, 80, 140, 130, 140, 30, 120, 30, 150, 225, 100, 170, 210],
    'Margin': [42000, 75360, 35200, 86800, 26390, 117880, 29370, 51000, 510, 20100, 166500, 2400, 52020, 65730],
    'Per Unit Rebate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dollor Investment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lower Discount': [lower_discount]*14,
    'Upper Discount': [upper_discount]*14,
    'Discount Uplift': [2.174962489, 1.174962489, 3.00034343, 1.174962489, 1.099384733, 2.174962489, 2.174962489, 2.974962489, 2.56904535, 1.625284225, 1.889298444, 3.038315124, 2.700573718, 5.485022623],
    'Tactic Uplift': [1]*14
}


# Create DataFrames
Tea = pd.DataFrame(data1)
Poha = pd.DataFrame(data2)
Coffee = pd.DataFrame(data3)
Spices = pd.DataFrame(data4)
Masala = pd.DataFrame(data5)

total = {
    'Data-Totals': ['Units', 'Revenue', 'Margins', 'Investment'],
    'Planned': [6114, 1797440, 771260, 0],
    'Optimized': [6114, 1797440, 771260, 0],
    '% Change': [0.0, 0.0, 0.0, 0.0]
}

df = pd.DataFrame(data1)
tf = pd.DataFrame(total)

st.title("KPI Optimization", anchor='Trade Optimization')

category_dict = {
    "Tea": Tea,
    "Poha": Poha,
    "Coffee": Coffee,
    "Spices": Spices,
    "Masala": Masala
}

# Fetch and display the data based on the selected option
df = category_dict.get(option, pd.DataFrame())

if st.button("Get Data"):
    st.write("Planned Data:")
    selected_columns = ['PL4 Sub-Category','Forecast Avg. Base Volume', 'Units', 'Revenue', 'Margin', 'Discount']
    edited_df = st.data_editor(df[selected_columns], key='planned_data_editor')
    st.session_state['edited_df'] = edited_df

# Optimize function
@st.cache_data
def optimize_discounts(df, max_investment, lower_discount, upper_discount):
    def objective(discounts):
        df['Optimized Discount'] = discounts
        optimized_units = df['Forecast Avg. Base Volume'] * np.exp(df['Discount Uplift'] * df['Optimized Discount'] / 100) * df['Tactic Uplift']
        revenue = optimized_units * (df['List/ Base Price'] - discounts)
        return -revenue.sum()

    x0 = np.full(len(df), (lower_discount + upper_discount) / 2)
    bounds = [(lower_discount, upper_discount)] * len(df)

    def investment_constraint(discounts):
        df['Optimized Discount'] = discounts
        optimized_units = df['Forecast Avg. Base Volume'] * np.exp(df['Discount Uplift'] * df['Optimized Discount'] / 100) * df['Tactic Uplift']
        investment = (df['Optimized Discount'] / 100) * df['List/ Base Price'] * optimized_units
        return max_investment - investment.sum()

    constraints = [{'type': 'ineq', 'fun': investment_constraint}]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='trust-constr', options={'disp': True})

    if result.success:
        df['Optimized Discount'] = np.round(result.x, 2)
        df['Optimized Units'] = df['Forecast Avg. Base Volume'] * np.exp(df['Discount Uplift'] * df['Optimized Discount'] / 100) * df['Tactic Uplift']
        df['Optimized Revenue'] = df['Optimized Units'] * (df['List/ Base Price'] - df['Optimized Discount'])
        df['Optimized Margin'] = df['Optimized Revenue'] - (df['Optimized Units'] * df['Per Unit COGS'])
        df['investment'] = (df['Optimized Discount'] / 100) * df['List/ Base Price'] * df['Optimized Units']
    else:
        st.error("Optimization failed. Please check constraints and data.")
        df = pd.DataFrame()  # Return an empty DataFrame on failure
    
    return df

if st.button("Optimize"):
    optimized_df = optimize_discounts(df, max_investment, lower_discount, upper_discount)

    if 'optimized_df' not in st.session_state:
        st.session_state['optimized_df'] = optimized_df

    total_units = st.session_state['optimized_df']['Optimized Units'].sum()
    total_revenue = st.session_state['optimized_df']['Optimized Revenue'].sum()
    total_margin = st.session_state['optimized_df']['Optimized Margin'].sum()
    total_investment = st.session_state['optimized_df']['investment'].sum()

    tf.loc[tf['Data-Totals'] == 'Units', 'Optimized'] = total_units
    tf.loc[tf['Data-Totals'] == 'Revenue', 'Optimized'] = total_revenue
    tf.loc[tf['Data-Totals'] == 'Margins', 'Optimized'] = total_margin
    tf.loc[tf['Data-Totals'] == 'Investment', 'Optimized'] = total_investment

    tf['% Change'] = ((tf['Optimized'] - tf['Planned']) / tf['Planned']) * 100
    st.write("Total Metrics:")
    st.dataframe(tf)

    selected_columns = ['PL4 Sub-Category','Forecast Avg. Base Volume', 'Optimized Units', 'Optimized Revenue', 'Optimized Margin', 'Optimized Discount']
    # optimized_df1= optimized_df[selected_columns]
    optimized_df1=st.session_state['optimized_df'][selected_columns]

    st.write("Optimized Data:")
    st.data_editor(optimized_df1, key='optimized_data_editor')
    
    # st.session_state['optimized_df'] = edited_optimized_df

    # Plot comparison chart
    st.write("Comparison of Planned and Optimized Values:")
    # comparison_data = tf.set_index('Data-Totals')[['Planned', 'Optimized']]
    # st.bar_chart(comparison_data)



    # Filter data for Units and Revenue
    planned_units = tf[(tf['Data-Totals'] == 'Units') & (tf['Planned'].notna())]
    optimized_units = tf[(tf['Data-Totals'] == 'Units') & (tf['Optimized'].notna())]
    planned_revenue = tf[(tf['Data-Totals'] == 'Revenue') & (tf['Planned'].notna())]
    optimized_revenue = tf[(tf['Data-Totals'] == 'Revenue') & (tf['Optimized'].notna())]

    # Prepare data
    metrics = ['Planned Units', 'Optimized Units', 'Planned Revenue', 'Optimized Revenue']
    units_values = [planned_units['Planned'].values[0], optimized_units['Optimized'].values[0]]
    revenue_values = [planned_revenue['Planned'].values[0], optimized_revenue['Optimized'].values[0]]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for Units
    color = 'tab:blue'
    bars_units = ax1.bar(['Planned Units', 'Optimized Units'], units_values, color=color, alpha=0.6, label='Units')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Units', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Add text labels on bars for Units
    for bar in bars_units:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

    # Create a second y-axis for Revenue
    ax2 = ax1.twinx()
    color = 'tab:green'
    bars_revenue = ax2.bar(['Planned Revenue', 'Optimized Revenue'], revenue_values, color=color, alpha=0.6, label='Revenue')
    ax2.set_ylabel('Revenue', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add text labels on bars for Revenue
    for bar in bars_revenue:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

    # Add title and adjust layout
    plt.title('Planned and Optimized Units and Revenue')
    fig.tight_layout()
    plt.grid(axis='y')

    # Show plot in Streamlit
    st.pyplot(fig)
# Add this section after the "Optimize" button
if 'optimized_df' in st.session_state:
    st.write("Current Optimized Data:")
    selected_columns = ['PL4 Sub-Category','Forecast Avg. Base Volume', 'Optimized Units', 'Optimized Revenue', 'Optimized Margin', 'Optimized Discount']
    edited_optimized_df=st.data_editor(st.session_state['optimized_df'][selected_columns])
    st.session_state['optimized_df'].loc[:, selected_columns]  = edited_optimized_df

if st.button("Process"):
    if 'optimized_df' in st.session_state:
        edf = st.session_state['optimized_df']

        edf['Units'] = edf['Forecast Avg. Base Volume'] * np.exp(edf['Discount Uplift'] * edf['Optimized Discount'] / 100) * edf['Tactic Uplift']
        edf['Revenue'] = edf['Units'] * (edf['Per Unit Selling Price'] - edf['Optimized Discount'])
        edf['Margin'] = edf['Revenue'] - (edf['Units'] * edf['Per Unit COGS'])
        edf['Dollor Investment'] = edf['Units'] * (edf['Optimized Discount'] / 100) * edf['List/ Base Price']

        st.write("Updated Planned Data:")
        selected_columns = ['PL4 Sub-Category','Forecast Avg. Base Volume', 'Optimized Units', 'Optimized Revenue', 'Optimized Margin', 'Optimized Discount']
        st.dataframe(edf[selected_columns])

        tf.at[0, 'Optimized'] = edf['Units'].sum()
        tf.at[1, 'Optimized'] = edf['Revenue'].sum()
        tf.at[2, 'Optimized'] = edf['Margin'].sum()
        tf.at[3, 'Optimized'] = edf['Dollor Investment'].sum()

        tf.at[0, 'Planned'] = df['Units'].sum()
        tf.at[1, 'Planned'] = df['Revenue'].sum()
        tf.at[2, 'Planned'] = df['Margin'].sum()
        tf.at[3, 'Planned'] = df['Dollor Investment'].sum()

        tf['% Change'] = ((tf['Optimized'] - tf['Planned']) / tf['Planned']) * 100

        st.write("Updated Total Comparison Data")
        st.dataframe(tf)

        # Plot updated comparison chart
        st.write("Updated Comparison of Planned and Optimized Values:")
        comparison_data_updated = tf.set_index('Data-Totals')[['Planned', 'Optimized']]
        st.bar_chart(comparison_data_updated)

        # Reuse the plot from the Optimize section
        st.write("Updated Comparison of Planned and Optimized Values (Units and Revenue):")

        planned_units = tf[(tf['Data-Totals'] == 'Units') & (tf['Planned'].notna())]
        optimized_units = tf[(tf['Data-Totals'] == 'Units') & (tf['Optimized'].notna())]
        planned_revenue = tf[(tf['Data-Totals'] == 'Revenue') & (tf['Planned'].notna())]
        optimized_revenue = tf[(tf['Data-Totals'] == 'Revenue') & (tf['Optimized'].notna())]

        units_values = [planned_units['Planned'].values[0], optimized_units['Optimized'].values[0]]
        revenue_values = [planned_revenue['Planned'].values[0], optimized_revenue['Optimized'].values[0]]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        bars_units = ax1.bar(['Planned Units', 'Optimized Units'], units_values, color=color, alpha=0.6, label='Units')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Units', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        for bar in bars_units:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

        ax2 = ax1.twinx()
        color = 'tab:green'
        bars_revenue = ax2.bar(['Planned Revenue', 'Optimized Revenue'], revenue_values, color=color, alpha=0.6, label='Revenue')
        ax2.set_ylabel('Revenue', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        for bar in bars_revenue:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

        plt.title('Updated Planned and Optimized Units and Revenue')
        fig.tight_layout()
        plt.grid(axis='y')

        st.pyplot(fig)

    else:
        st.write("No optimized data available. Please run optimization first.")
