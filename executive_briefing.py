import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

# ==================== DATA LOADING & CLEANING ====================

@st.cache_data
def load_state_caps():
    """Load and clean state emission caps and regulation data"""

    # Load Carbon Pricing Table
    state_caps_raw = pd.read_csv('Carbon_Pricing_Table.csv')

    # Extract the top section with detailed regulation info (rows with Jurisdiction)
    # Filter out rows where Jurisdiction is NaN
    state_caps = state_caps_raw[state_caps_raw['Jurisdiction'].notna()].copy()

    # Rename Jurisdiction to State for consistency
    state_caps = state_caps.rename(columns={'Jurisdiction': 'State'})

    # Select relevant columns - handle potential missing columns gracefully
    available_cols = state_caps.columns.tolist()
    required_cols = [
        'State',
        'Main Rule(s)',
        'Emissions_Threshold_tCO2',
        'Penalty_per_ton',
        'Who Is Covered / Thresholds',
        'Key Metric / Standard',
        'Cost if You Exceed / Don\'t Comply'
    ]

    # Only select columns that exist
    cols_to_select = [col for col in required_cols if col in available_cols]
    state_caps_clean = state_caps[cols_to_select].copy()

    # Remove any rows with all NaN (except for State column which should always have a value)
    state_caps_clean = state_caps_clean.dropna(how='all', subset=[col for col in cols_to_select if col != 'State'])

    return state_caps_clean

@st.cache_data
def load_and_prepare_data():
    """Load and clean all data from final_notebook analysis"""

    # Load electricity prices
    electricity_price = pd.read_csv('average_retail_price_of_electricity_annual.csv', header=4)
    df_2023 = electricity_price[electricity_price['Year'] == 2023]
    cols_industrial = [col for col in df_2023.columns if 'industrial cents per kilowatthour' in col]
    df_2023_industrial = df_2023[['Year'] + cols_industrial]

    melted = df_2023_industrial.melt(
        id_vars=['Year'],
        value_vars=cols_industrial,
        var_name='Region_Sector',
        value_name='cents_per_kWh'
    )

    def extract_state(colname):
        match = re.match(r'(.*) industrial cents per kilowatthour', colname)
        return match.group(1).strip() if match else colname

    melted['State'] = melted['Region_Sector'].apply(extract_state)
    melted['Grid_Price_per_MWh'] = melted['cents_per_kWh'] * 10
    clean_electric_prices = melted[['Year', 'State', 'cents_per_kWh', 'Grid_Price_per_MWh']]

    # Filter for actual US states
    us_state_names = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
        'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
        'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
        'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
        'Wisconsin', 'Wyoming'
    ]
    clean_electric_prices = clean_electric_prices[clean_electric_prices['State'].isin(us_state_names)].copy()

    # Load fuel costs
    fuel_price = pd.read_csv('Average_cost_of_fossil_fuels_for_electricity_generation_natural_gas_all_sectors_annual.csv')
    fuel_2023 = fuel_price[fuel_price['Year'] == 2023]

    melted_fuel = fuel_2023.melt(
        id_vars=['Year'],
        var_name='Column_Desc',
        value_name='Gas_Cost_per_MMBtu'
    )

    def extract_state_fuel(col):
        match = re.match(r'(.*) dollars per mcf', col)
        return match.group(1).strip() if match else col

    melted_fuel['State'] = melted_fuel['Column_Desc'].apply(extract_state_fuel)
    clean_fuel_cost = melted_fuel[['Year', 'State', 'Gas_Cost_per_MMBtu']]
    clean_fuel_cost = clean_fuel_cost[clean_fuel_cost['State'].isin(us_state_names)].copy()

    # Load facility data
    facility_data = pd.read_csv('EIA923_Schedules_2_3_4_5_M_12_2023_Final_Revision.csv')
    ng_plants = facility_data[facility_data['Reported\nFuel Type Code'] == 'NG'].copy()

    ng_plants['Total Fuel Consumption\nMMBtu'] = pd.to_numeric(
        ng_plants['Total Fuel Consumption\nMMBtu'].astype(str).str.replace(',', ''),
        errors='coerce'
    )
    ng_plants['Net Generation\n(Megawatthours)'] = pd.to_numeric(
        ng_plants['Net Generation\n(Megawatthours)'].astype(str).str.replace(',', ''),
        errors='coerce'
    )

    clean_plants = (
        ng_plants.groupby(['Plant Id', 'Plant Name', 'Plant State'], as_index=False)
        .agg({
            'Total Fuel Consumption\nMMBtu': 'sum',
            'Net Generation\n(Megawatthours)': 'sum'
        })
    )

    # Map state abbreviations to full names
    state_map = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
        'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
        'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
        'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
        'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
        'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }

    clean_plants['Plant State'] = clean_plants['Plant State'].apply(lambda x: state_map.get(x, x))

    # Merge all data
    merged_1 = pd.merge(
        clean_plants,
        clean_fuel_cost.rename(columns={'State': 'Plant State'}),
        on='Plant State',
        how='inner'
    )

    master_analysis_table = pd.merge(
        merged_1,
        clean_electric_prices.rename(columns={'State': 'Plant State'}),
        on='Plant State',
        how='inner'
    )

    # Calculate operational metrics
    master_analysis_table['Operational_Cost'] = (
        master_analysis_table['Total Fuel Consumption\nMMBtu'] *
        master_analysis_table['Gas_Cost_per_MMBtu']
    )

    master_analysis_table['Net Revenue'] = (
        master_analysis_table['Net Generation\n(Megawatthours)'] *
        master_analysis_table['Grid_Price_per_MWh']
    )

    # Filter out zero values
    filtered_master = master_analysis_table[
        (master_analysis_table['Operational_Cost'] > 0) &
        (master_analysis_table['Net Revenue'] > 0)
    ].copy()

    # Convert to millions USD
    filtered_master['Generation_Cost_Million_USD'] = (filtered_master['Operational_Cost'] / 1_000_000).round(2)
    filtered_master['Net_Revenue_Million_USD'] = (filtered_master['Net Revenue'] / 1_000_000).round(2)

    # Calculate Net Operating Profit
    filtered_master['Net_Operating_Profit'] = (
        filtered_master['Net_Revenue_Million_USD'] -
        filtered_master['Generation_Cost_Million_USD']
    )

    # Calculate efficiency metrics
    filtered_master['Build_Cost_per_MWh'] = (
        filtered_master['Operational_Cost'] /
        filtered_master['Net Generation\n(Megawatthours)']
    )
    filtered_master['Efficiency_Gap'] = (
        filtered_master['Build_Cost_per_MWh'] -
        filtered_master['Grid_Price_per_MWh']
    )

    return filtered_master


# ==================== SIDEBAR: GLOBAL ASSUMPTIONS ====================

def create_sidebar_controls():
    """Create and manage global assumption sliders"""

    st.sidebar.title(" Portfolio 'What-If' Engine")
    st.sidebar.write("Adjust macro-economic assumptions to model portfolio resilience")

    # Gas Price Adjustment
    gas_adjustment = st.sidebar.slider(
        "Global Gas Cost Adjustment (%)",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
        help="Adjust natural gas costs across all plants"
    )

    # Grid Price Adjustment
    grid_adjustment = st.sidebar.slider(
        "Global Grid Price Adjustment (%)",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
        help="Adjust electricity prices across all markets"
    )

    # Generation Adjustment
    generation_adjustment = st.sidebar.slider(
        "Global Generation Adjustment (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Model demand changes or generation constraints"
    )

    return gas_adjustment, grid_adjustment, generation_adjustment


def apply_what_if_adjustments(df, gas_pct, grid_pct, gen_pct):
    """Apply what-if adjustments to the dataset"""

    df_adjusted = df.copy()

    # Apply gas cost adjustment
    df_adjusted['Gas_Cost_per_MMBtu'] = df_adjusted['Gas_Cost_per_MMBtu'] * (1 + gas_pct / 100)

    # Recalculate operational cost
    df_adjusted['Operational_Cost'] = (
        df_adjusted['Total Fuel Consumption\nMMBtu'] *
        df_adjusted['Gas_Cost_per_MMBtu']
    )

    # Apply grid price adjustment
    df_adjusted['Grid_Price_per_MWh'] = df_adjusted['Grid_Price_per_MWh'] * (1 + grid_pct / 100)

    # Apply generation adjustment
    df_adjusted['Net Generation\n(Megawatthours)'] = (
        df_adjusted['Net Generation\n(Megawatthours)'] * (1 + gen_pct / 100)
    )

    # Recalculate net revenue
    df_adjusted['Net Revenue'] = (
        df_adjusted['Net Generation\n(Megawatthours)'] *
        df_adjusted['Grid_Price_per_MWh']
    )

    # Recalculate financial metrics
    df_adjusted['Generation_Cost_Million_USD'] = (df_adjusted['Operational_Cost'] / 1_000_000).round(2)
    df_adjusted['Net_Revenue_Million_USD'] = (df_adjusted['Net Revenue'] / 1_000_000).round(2)
    df_adjusted['Net_Operating_Profit'] = (
        df_adjusted['Net_Revenue_Million_USD'] -
        df_adjusted['Generation_Cost_Million_USD']
    )

    # Recalculate efficiency metrics
    df_adjusted['Build_Cost_per_MWh'] = (
        df_adjusted['Operational_Cost'] /
        df_adjusted['Net Generation\n(Megawatthours)']
    )
    df_adjusted['Efficiency_Gap'] = (
        df_adjusted['Build_Cost_per_MWh'] -
        df_adjusted['Grid_Price_per_MWh']
    )

    return df_adjusted


# ==================== HEADLINE NUMBERS (KPIs) ====================

def display_headline_kpis(df):
    """Display top-level KPI metrics"""

    st.header(" Executive Briefing Dashboard")

    # Calculate KPIs
    total_net_operating_profit = df['Net_Operating_Profit'].sum()

    # Cost of Inefficiency = sum of negative profits
    cost_of_inefficiency = abs(df[df['Net_Operating_Profit'] < 0]['Net_Operating_Profit'].sum())

    unprofitable_count = (df['Net_Operating_Profit'] < 0).sum()
    total_plants = len(df)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Net Operating Profit",
            f"${total_net_operating_profit:,.0f}M",
            delta=f"from {total_plants} plants" if total_net_operating_profit > 0 else "NEGATIVE"
        )

    with col2:
        st.metric(
            "Cost of Inefficiency",
            f"${cost_of_inefficiency:,.0f}M",
            delta=f"from {unprofitable_count} plants"
        )

    with col3:
        st.metric(
            "Unprofitable Plants",
            f"{unprofitable_count} of {total_plants}",
            delta=f"{(unprofitable_count/total_plants*100):.1f}%"
        )

    with col4:
        profitable_pct = ((total_plants - unprofitable_count) / total_plants * 100)
        st.metric(
            "Profitable Plants",
            f"{(total_plants - unprofitable_count)} of {total_plants}",
            delta=f"{profitable_pct:.1f}%"
        )

    st.divider()


# ==================== PROFIT POOL & DRAIN (STATE-LEVEL P&L) ====================

def display_state_level_analysis(df):
    """Display state-level profit and loss analysis"""

    st.header("Profit Pool & Drain by State")

    # Aggregate by state
    state_profit = (
        df.groupby('Plant State')['Net_Operating_Profit']
        .sum()
        .reset_index()
        .sort_values('Net_Operating_Profit', ascending=False)
    )

    # Create two columns for context
    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart
        st.bar_chart(
            data=state_profit.set_index('Plant State')['Net_Operating_Profit'],
            use_container_width=True,
            height=400
        )

    with col2:
        st.subheader("Top 5 Performers")
        top5 = state_profit.head(5)
        for idx, row in top5.iterrows():
            st.write(f"**{row['Plant State']}** | ${row['Net_Operating_Profit']:,.0f}M")

        st.divider()

        st.subheader("Bottom 5 Drains")
        bottom5 = state_profit.tail(5)
        for idx, row in bottom5.iterrows():
            st.write(f"**{row['Plant State']}** | ${row['Net_Operating_Profit']:,.0f}M")

    st.divider()


# ==================== ACTION LIST: TOP 10 LOSERS ====================

def display_top_10_losers(df):
    """Display dynamic Top 10 worst-performing plants"""

    st.header(" Action List: Top 10 Worst Performers")

    # Sort by Net Operating Profit (ascending = worst first)
    top_losers = df.nsmallest(10, 'Net_Operating_Profit')

    # Prepare display columns
    display_cols = [
        'Plant Name',
        'Plant State',
        'Net_Operating_Profit',
        'Generation_Cost_Million_USD',
        'Net_Revenue_Million_USD',
        'Build_Cost_per_MWh',
        'Grid_Price_per_MWh'
    ]

    display_df = top_losers[display_cols].copy()
    display_df.columns = [
        'Plant Name',
        'State',
        'Net Operating Profit ($M)',
        'Gen Cost ($M)',
        'Revenue ($M)',
        'Build Cost ($/MWh)',
        'Grid Price ($/MWh)'
    ]

    # Add zombie plant identifier
    display_df['Status'] = top_losers['Net Generation\n(Megawatthours)'].apply(
        lambda x: ' ZOMBIE' if x == 0 else ' LOSING'
    )

    st.dataframe(
        display_df.style.format({
            'Net Operating Profit ($M)': '{:,.1f}',
            'Gen Cost ($M)': '{:,.1f}',
            'Revenue ($M)': '{:,.1f}',
            'Build Cost ($/MWh)': '{:,.1f}',
            'Grid Price ($/MWh)': '{:,.1f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()


# ==================== PLANT DEEP-DIVE ====================

def display_plant_deep_dive(df, state_caps):
    """Interactive plant deep-dive analysis with state emission caps context"""

    st.header("Plant Deep-Dive Analysis")

    # Create filter options for users to narrow down plant selection
    col1, col2, col3 = st.columns(3)

    with col1:
        # Filter by state
        available_states = sorted(df['Plant State'].unique().tolist())
        selected_state_filter = st.selectbox(
            "Filter by State (optional):",
            options=['All States'] + available_states,
            key="state_filter"
        )

    with col2:
        # Filter by profitability
        profitability_filter = st.selectbox(
            "Filter by Profitability:",
            options=[
                'All Plants',
                'Profitable Only',
                'Unprofitable Only',
                'Zombie Plants Only'
            ],
            key="profit_filter"
        )

    with col3:
        # Sort option
        sort_option = st.selectbox(
            "Sort by:",
            options=[
                'Profitability (Worst First)',
                'Profitability (Best First)',
                'Plant Name (A-Z)',
                'Generation (High to Low)',
                'Profit/Loss (Highest)',
                'Cost of Inefficiency'
            ],
            key="sort_filter"
        )

    # Apply filters
    filtered_df = df.copy()

    # State filter
    if selected_state_filter != 'All States':
        filtered_df = filtered_df[filtered_df['Plant State'] == selected_state_filter]

    # Profitability filter
    if profitability_filter == 'Profitable Only':
        filtered_df = filtered_df[filtered_df['Net_Operating_Profit'] > 0]
    elif profitability_filter == 'Unprofitable Only':
        filtered_df = filtered_df[filtered_df['Net_Operating_Profit'] < 0]
    elif profitability_filter == 'Zombie Plants Only':
        filtered_df = filtered_df[filtered_df['Net Generation\n(Megawatthours)'] == 0]

    # Apply sorting
    if sort_option == 'Profitability (Worst First)':
        filtered_df = filtered_df.sort_values('Net_Operating_Profit', ascending=True)
    elif sort_option == 'Profitability (Best First)':
        filtered_df = filtered_df.sort_values('Net_Operating_Profit', ascending=False)
    elif sort_option == 'Plant Name (A-Z)':
        filtered_df = filtered_df.sort_values('Plant Name', ascending=True)
    elif sort_option == 'Generation (High to Low)':
        filtered_df = filtered_df.sort_values('Net Generation\n(Megawatthours)', ascending=False)
    elif sort_option == 'Profit/Loss (Highest)':
        filtered_df = filtered_df.sort_values('Net Revenue (Million USD)', ascending=False)
    elif sort_option == 'Cost of Inefficiency':
        filtered_df = filtered_df.sort_values('Generation_Cost_Million_USD', ascending=False)

    # Create selectbox with all plants (filtered)
    plant_options = []
    for idx, row in filtered_df.iterrows():
        zombie_flag = "" if row['Net Generation\n(Megawatthours)'] == 0 else ""
        profit_indicator = "📈" if row['Net_Operating_Profit'] > 0 else "📉"
        label = f"{zombie_flag}{profit_indicator} {row['Plant Name']} ({row['Plant State']}) - ${row['Net_Operating_Profit']:.1f}M"
        plant_options.append((label, idx))

    st.write(f"**Showing {len(plant_options)} of {len(df)} plants**")

    if len(plant_options) == 0:
        st.warning("No plants match your filters. Try adjusting them.")
        return

    # Create selectbox
    selected_plant_label = st.selectbox(
        "Select a plant to analyze:",
        options=[opt[0] for opt in plant_options],
        key="plant_selector"
    )

    # Find the corresponding plant using the original index
    selected_label_idx = [opt[0] for opt in plant_options].index(selected_plant_label)
    selected_plant_original_idx = plant_options[selected_label_idx][1]
    selected_plant = df.loc[selected_plant_original_idx]

    # Display plant details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f" {selected_plant['Plant Name']}")
        st.write(f"**State:** {selected_plant['Plant State']}")
        st.write(f"**Net Generation:** {selected_plant['Net Generation\n(Megawatthours)']:,.0f} MWh")
        st.write(f"**Total Fuel Consumption:** {selected_plant['Total Fuel Consumption\nMMBtu']:,.0f} MMBtu")

    with col2:
        st.subheader("Financial Snapshot")
        st.write(f"**Generation Cost:** ${selected_plant['Generation_Cost_Million_USD']:.2f}M")
        st.write(f"**Net Revenue:** ${selected_plant['Net_Revenue_Million_USD']:.2f}M")
        st.write(f"**Net Operating Profit:** ${selected_plant['Net_Operating_Profit']:.2f}M")

    with col3:
        # Add a couple of empty lines to vertically align the button
        st.write(" ") 
        st.write(" ")
        # Create the button. The key is VITAL to make it unique for each plant!
        show_image = st.button("View Facility Image", key=f"img_btn_{selected_plant['Plant Id']}")

    # If the button is clicked, 'show_image' becomes True for this run
    if show_image:
        
        # --- IMPORTANT ---
        # Replace this URL with your actual image path.
        # It can be a local file (e.g., "images/plant_123.jpg")
        # or a URL. I am using a placeholder.
        
        placeholder_image = "plant123.jpg"
        
        st.image(
            placeholder_image,
            caption=f"Plant Process Health for {selected_plant['Plant Name']}",
            use_column_width=True
        )

    # Display State Emission Caps if applicable
    plant_state = selected_plant['Plant State']
    state_cap_info = state_caps[state_caps['State'] == plant_state]

    if not state_cap_info.empty:
        st.divider()
        st.subheader(f"⚖️ State Emission Regulations: {plant_state}")

        cap_row = state_cap_info.iloc[0]

        # Create two-column layout for regulation details
        col1, col2 = st.columns(2)

        with col1:
            if 'Main Rule(s)' in state_caps.columns:
                st.write("**Regulation Name:**")
                st.write(cap_row['Main Rule(s)'])

            if 'Emissions_Threshold_tCO2' in state_caps.columns:
                st.write("**Emission Threshold:**")
                threshold = cap_row['Emissions_Threshold_tCO2']
                if pd.notna(threshold):
                    st.write(f"{float(threshold):,.0f} tCO2")

            if 'Penalty_per_ton' in state_caps.columns:
                st.write("**Penalty per Ton:**")
                penalty = cap_row['Penalty_per_ton']
                if pd.notna(penalty):
                    st.write(f"${float(penalty)}")

        with col2:
            if 'Who Is Covered / Thresholds' in state_caps.columns:
                st.write("**Coverage & Thresholds:**")
                coverage = cap_row['Who Is Covered / Thresholds']
                if pd.notna(coverage):
                    st.write(coverage)

            if 'Key Metric / Standard' in state_caps.columns:
                st.write("**Key Metric/Standard:**")
                metric = cap_row['Key Metric / Standard']
                if pd.notna(metric):
                    st.write(metric)

        if 'Cost if You Exceed / Don\'t Comply' in state_caps.columns:
            cost_info = cap_row['Cost if You Exceed / Don\'t Comply']
            if pd.notna(cost_info):
                st.write("**Compliance Cost if Exceeded:**")
                st.warning(cost_info)

        # Special context for Zombie Plants
        if selected_plant['Net Generation\n(Megawatthours)'] == 0:
            if 'Emissions_Threshold_tCO2' in state_caps.columns:
                threshold = cap_row['Emissions_Threshold_tCO2']
                if pd.notna(threshold):
                    st.info(
                        f"**ZOMBIE PLANT CONTEXT**: This plant produced zero MWh in 2023, so it likely falls below "
                        f"the {plant_state} emission threshold of {float(threshold):,.0f} tCO2. "
                        f"However, it still incurred generation costs, making it an extreme operational loss."
                    )
                else:
                    st.info(
                        f"**ZOMBIE PLANT CONTEXT**: This plant produced zero MWh in 2023, generating zero emissions. "
                        f"However, it still incurred generation costs, making it an extreme operational loss."
                    )

    # Build vs Buy Analysis
    st.subheader("📊 Build vs. Buy Analysis")

    build_cost = selected_plant['Build_Cost_per_MWh']
    grid_price = selected_plant['Grid_Price_per_MWh']

    # Only show if not infinity (zombie plant)
    if np.isfinite(build_cost):
        col1, col2 = st.columns(2)

        with col1:
            # Simple comparison chart data
            comparison_data = pd.DataFrame({
                'Metric': ['Build Cost\nper MWh', 'Grid Price\nper MWh'],
                'Cost ($/MWh)': [build_cost, grid_price]
            })

            st.bar_chart(
                data=comparison_data.set_index('Metric'),
                use_container_width=True,
                height=300
            )

        with col2:
            st.metric(
                "Efficiency Gap",
                f"${build_cost - grid_price:.2f}/MWh",
                delta="NEGATIVE (Unprofitable)" if build_cost > grid_price else "POSITIVE (Profitable)"
            )
    else:
        st.warning(" **ZOMBIE PLANT**: This plant produced zero MWh. Build cost cannot be calculated.")

    # Local Efficiency Improvement Toggle
    st.subheader("🔧 Local Efficiency Improvement Scenario")

    efficiency_gain_pct = st.slider(
        "Local Efficiency Gain (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="Reduce generation cost through efficiency improvements"
    )

    if efficiency_gain_pct > 0:
        new_generation_cost = selected_plant['Generation_Cost_Million_USD'] * (1 - efficiency_gain_pct / 100)
        new_net_operating_profit = selected_plant['Net_Revenue_Million_USD'] - new_generation_cost

        improvement = new_net_operating_profit - selected_plant['Net_Operating_Profit']

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "New Generation Cost",
                f"${new_generation_cost:.2f}M",
                delta=f"-${selected_plant['Generation_Cost_Million_USD'] - new_generation_cost:.2f}M"
            )

        with col2:
            st.metric(
                "Improved Net Operating Profit",
                f"${new_net_operating_profit:.2f}M",
                delta=f"+${improvement:.2f}M" if improvement > 0 else f"-${abs(improvement):.2f}M"
            )

        # Analysis
        if new_net_operating_profit > 0:
            st.success(
                f" **With {efficiency_gain_pct}% efficiency gain, this plant becomes profitable!** "
                f"Target: Reduce generation costs by ${selected_plant['Generation_Cost_Million_USD'] - new_generation_cost:.2f}M"
            )
        else:
            st.info(
                f"ℹ️ **An additional {abs(new_net_operating_profit):.2f}M in efficiency gains is needed** "
                f"to reach breakeven."
            )

    st.divider()


# ==================== MAIN APP ====================

def main():
    st.set_page_config(
        page_title="Executive Briefing Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load data
    df = load_and_prepare_data()
    state_caps = load_state_caps()

    # Sidebar controls
    gas_adj, grid_adj, gen_adj = create_sidebar_controls()

    # Apply what-if adjustments
    df_adjusted = apply_what_if_adjustments(df, gas_adj, grid_adj, gen_adj)

    # Display adjustment summary if any adjustments made
    if gas_adj != 0 or grid_adj != 0 or gen_adj != 0:
        st.info(
            f" **What-If Scenario Active:**\n"
            f"Gas +{gas_adj}% | Grid {grid_adj:+d}% | Generation {gen_adj:+d}% "
        )

    # Main display sections
    display_headline_kpis(df_adjusted)
    display_state_level_analysis(df_adjusted)
    display_top_10_losers(df_adjusted)
    display_plant_deep_dive(df_adjusted, state_caps)


if __name__ == "__main__":
    main()
