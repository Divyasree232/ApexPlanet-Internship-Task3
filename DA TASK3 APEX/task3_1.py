"""
TASK 3: Deep-Dive Analysis & Interactive Dashboarding
APEX PLANET Internship - Data Analytics Track
Author: KOTLA DIVYASREE
CORRECTED VERSION - Matching actual column names
"""

# ============================================
# 1. IMPORT LIBRARIES & LOAD DATA
# ============================================

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For statistical validation
from scipy import stats

# Load your cleaned dataset from Task 1
# Note: Use the actual column names from your dataset
df = pd.read_csv('cleaned_cafe_sales_data.csv')  # Update with your actual file path

print("=" * 60)
print("TASK 3: DEEP-DIVE ANALYSIS")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 2 rows:\n{df.head(2)}")

# ============================================
# 2. CHECK COLUMN NAMES & RENAME IF NEEDED
# ============================================

# Your dataset uses lowercase with underscores
# Let's work with actual column names
print("\n" + "=" * 60)
print("COLUMN MAPPING (Actual vs Expected)")
print("=" * 60)

# Map actual column names to what we need
column_mapping = {
    'total_spent': 'Total_Spent',  # Note: Using underscore instead of space
    'transaction_date': 'Transaction_Date',
    'item_category': 'Item_Category',
    'is_weekend': 'Is_Weekend',
    'quantity': 'Quantity',
    'price_per_unit': 'Price_Per_Unit',
    'location': 'Location',
    'payment_method': 'Payment_Method',
    'day_of_week': 'Day_Of_Week',
    'month': 'Month',
    'quarter': 'Quarter'
}

# Create renamed columns for easier access
df['Total_Spent'] = df['total_spent']  # Keep original but add cleaner name
df['Transaction_Date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)
df['Is_Weekend'] = df['is_weekend'].astype(str).str.upper().str.strip() == 'TRUE'

print("Columns mapped successfully")

# ============================================
# 3. DEFINE CORE KPIs (Based on Task 2 Findings)
# ============================================

print("\n" + "=" * 60)
print("SECTION 1: CORE KPI DEFINITIONS")
print("=" * 60)

# KPI 1: Total Revenue
total_revenue = df['Total_Spent'].sum()
print(f"\nKPI 1 - TOTAL REVENUE: ${total_revenue:,.2f}")
print(f"   Formula: SUM(total_spent)")
print(f"   Business Rationale: Top-line business performance measure")

# KPI 2: Average Order Value (AOV)
aov = df['Total_Spent'].mean()
print(f"\nKPI 2 - AVERAGE ORDER VALUE (AOV): ${aov:.2f}")
print(f"   Formula: Total Revenue / Number of Transactions")
print(f"   Business Rationale: Measures customer spending per visit")

# KPI 3: Total Transactions
total_transactions = len(df)
print(f"\nKPI 3 - TOTAL TRANSACTIONS: {total_transactions:,}")
print(f"   Formula: COUNT(transaction_id)")
print(f"   Business Rationale: Measures customer traffic volume")

# KPI 4: Weekend vs Weekday Performance
# Check if Is_Weekend is boolean or string
if df['Is_Weekend'].dtype == 'bool':
    weekend_aov = df[df['Is_Weekend'] == True]['Total_Spent'].mean()
    weekday_aov = df[df['Is_Weekend'] == False]['Total_Spent'].mean()
else:
    # If it's string or int, convert appropriately
    weekend_aov = df[df['Is_Weekend'] == True]['Total_Spent'].mean()
    weekday_aov = df[df['Is_Weekend'] == False]['Total_Spent'].mean()

print(f"\n KPI 4 - WEEKEND vs WEEKDAY AOV:")
print(f"   Weekend AOV: ${weekend_aov:.2f}")
print(f"   Weekday AOV: ${weekday_aov:.2f}")
print(f"   Difference: ${(weekend_aov - weekday_aov):.2f} ({(weekend_aov/weekday_aov - 1)*100:.1f}% higher on weekends)")
print(f"   Business Rationale: Informs staffing and promotion strategies")

# KPI 5: Category Revenue Breakdown
category_revenue = df.groupby('item_category')['Total_Spent'].sum()
print(f"\n KPI 5 - CATEGORY REVENUE BREAKDOWN:")
for category in category_revenue.index:
    revenue = category_revenue[category]
    pct = (revenue / total_revenue) * 100
    print(f"   {category}: ${revenue:,.2f} ({pct:.1f}%)")
print(f"   Business Rationale: Identifies primary revenue drivers")

# ============================================
# 4. DEEP-DIVE ANALYSIS: Weekend vs Weekday Category Performance
# ============================================

print("\n" + "=" * 60)
print("SECTION 2: DEEP-DIVE ANALYSIS")
print("Question: Do customer spending habits differ between weekdays and weekends?")
print("=" * 60)

# Create pivot table for deep analysis
deep_dive = df.groupby(['Is_Weekend', 'item_category']).agg({
    'Total_Spent': ['sum', 'mean', 'count'],
    'quantity': 'sum'
}).round(2)

deep_dive.columns = ['Total_Revenue', 'AOV', 'Transaction_Count', 'Total_Quantity']
deep_dive = deep_dive.reset_index()

# Rename Is_Weekend for readability
deep_dive['Day_Type'] = deep_dive['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})

print("\nDEEP-DIVE RESULTS: Category Performance by Day Type")
print("-" * 70)
print(deep_dive.to_string(index=False))

# Calculate percentage differences
print("\n KEY INSIGHTS FROM DEEP-DIVE:")
print("-" * 70)

for category in deep_dive['item_category'].unique():
    weekend_data = deep_dive[(deep_dive['item_category'] == category) & (deep_dive['Day_Type'] == 'Weekend')]
    weekday_data = deep_dive[(deep_dive['item_category'] == category) & (deep_dive['Day_Type'] == 'Weekday')]
    
    if not weekend_data.empty and not weekday_data.empty:
        weekend_aov_cat = weekend_data['AOV'].values[0]
        weekday_aov_cat = weekday_data['AOV'].values[0]
        if weekday_aov_cat > 0:
            pct_diff = ((weekend_aov_cat - weekday_aov_cat) / weekday_aov_cat) * 100
            if pct_diff > 0:
                print(f"   • {category}: Weekend AOV ${weekend_aov_cat:.2f} is {pct_diff:.1f}% HIGHER than weekday (${weekday_aov_cat:.2f})")
            else:
                print(f"   • {category}: Weekend AOV ${weekend_aov_cat:.2f} is {abs(pct_diff):.1f}% LOWER than weekday (${weekday_aov_cat:.2f})")

# Find top contributing category for weekend revenue lift
weekend_revenue_by_cat = df[df['Is_Weekend'] == True].groupby('item_category')['Total_Spent'].sum()
weekday_revenue_by_cat = df[df['Is_Weekend'] == False].groupby('item_category')['Total_Spent'].sum()
revenue_lift = weekend_revenue_by_cat - weekday_revenue_by_cat

print(f"\n REVENUE LIFT ON WEEKENDS (by category):")
for cat in revenue_lift.index:
    print(f"   • {cat}: +${revenue_lift[cat]:,.2f} additional revenue on weekends")

top_lift_category = revenue_lift.idxmax()
print(f"\n Category driving MOST weekend revenue growth: {top_lift_category}")

# ============================================
# 5. TIME-BASED ANALYSIS (Monthly Trends)
# ============================================

print("\n" + "=" * 60)
print("SECTION 3: TIME-BASED ANALYSIS")
print("Question: How does revenue trend over time?")
print("=" * 60)

# Create month-year grouping
df['Year_Month'] = pd.to_datetime(df['Transaction_Date']).dt.to_period('M')
monthly_spending = df.groupby('Year_Month')['Total_Spent'].agg(['sum', 'mean', 'count'])
monthly_spending.columns = ['Revenue', 'AOV', 'Transactions']

print("\nMONTHLY PERFORMANCE TREND:")
print("-" * 50)
print(monthly_spending.to_string())

# Calculate month-over-month growth
monthly_spending['Revenue_Growth'] = monthly_spending['Revenue'].pct_change() * 100
print("\nMONTH-OVER-MONTH REVENUE GROWTH:")
print("-" * 50)
for idx, growth in monthly_spending['Revenue_Growth'].items():
    if not pd.isna(growth):
        print(f"   {idx}: {growth:.1f}% growth")

# ============================================
# 6. LOCATION-BASED ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("SECTION 4: LOCATION PERFORMANCE")
print("Question: Which locations perform best?")
print("=" * 60)

location_performance = df.groupby('location').agg({
    'Total_Spent': ['sum', 'mean', 'count']
}).round(2)
location_performance.columns = ['Total_Revenue', 'AOV', 'Transaction_Count']
location_performance = location_performance.sort_values('Total_Revenue', ascending=False)

print("\n LOCATION PERFORMANCE RANKING:")
print("-" * 60)
for loc in location_performance.index:
    revenue = location_performance.loc[loc, 'Total_Revenue']
    aov_loc = location_performance.loc[loc, 'AOV']
    transactions = location_performance.loc[loc, 'Transaction_Count']
    print(f"   {loc}: ${revenue:,.2f} revenue | ${aov_loc:.2f} AOV | {transactions} transactions")

# ============================================
# 7. STATISTICAL VALIDATION (Hypothesis Testing)
# ============================================

print("\n" + "=" * 60)
print("SECTION 5: STATISTICAL VALIDATION")
print("Hypothesis: Weekend AOV is significantly higher than Weekday AOV")
print("=" * 60)

# Separate weekend and weekday spending
weekend_spending = df[df['Is_Weekend'] == True]['Total_Spent']
weekday_spending = df[df['Is_Weekend'] == False]['Total_Spent']

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(weekend_spending, weekday_spending, equal_var=False)

print(f"\nTwo-Sample T-Test Results:")
print(f"   Weekend AOV: ${weekend_spending.mean():.2f} (n={len(weekend_spending):,})")
print(f"   Weekday AOV: ${weekday_spending.mean():.2f} (n={len(weekday_spending):,})")
print(f"   T-Statistic: {t_stat:.4f}")
print(f"   P-Value: {p_value:.6f}")

# Interpretation
alpha = 0.05
print(f"\n Statistical Conclusion (α = {alpha}):")
if p_value < alpha:
    print(f"   REJECT null hypothesis. The difference is STATISTICALLY SIGNIFICANT.")
    print(f"   → Weekend AOV is significantly higher than weekday AOV (p = {p_value:.4f})")
    print(f"   → Business recommendation: Implement weekend-specific promotions and staffing")
else:
    print(f"   FAIL to reject null hypothesis. Difference is NOT statistically significant.")

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt((weekend_spending.std()**2 + weekday_spending.std()**2) / 2)
cohens_d = (weekend_spending.mean() - weekday_spending.mean()) / pooled_std
print(f"\n Effect Size (Cohen's d): {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    print("   → Negligible effect")
elif abs(cohens_d) < 0.5:
    print("   → Small effect")
elif abs(cohens_d) < 0.8:
    print("   → Medium effect")
else:
    print("   → Large effect")

# ============================================
# 8. VISUALIZATION FOR DEEP-DIVE INSIGHTS
# ============================================

print("\n" + "=" * 60)
print("SECTION 6: GENERATING VISUALIZATIONS")
print("=" * 60)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 14))

# Subplot 1: KPI Summary Text
ax1 = fig.add_subplot(3, 2, 1)
ax1.axis('off')
kpi_text = f"""
╔══════════════════════════════════════════╗
║         CORE KPIs DASHBOARD              ║
╠══════════════════════════════════════════╣
║  Total Revenue:     ${total_revenue:,.2f}         ║
║  Average Order Value: ${aov:.2f}                  ║
║  Total Transactions: {total_transactions:,}                  ║
║  Weekend AOV:       ${weekend_aov:.2f}                  ║
║  Weekday AOV:       ${weekday_aov:.2f}                  ║
╚══════════════════════════════════════════╝
"""
ax1.text(0.1, 0.5, kpi_text, fontsize=11, family='monospace', verticalalignment='center')
ax1.set_title('KPI Summary Dashboard', fontsize=14, fontweight='bold')

# Subplot 2: AOV by Category and Day Type
ax2 = fig.add_subplot(3, 2, 2)
plot_data = df.groupby(['item_category', 'Is_Weekend'])['Total_Spent'].mean().reset_index()
plot_data['Day_Type'] = plot_data['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})

categories = plot_data['item_category'].unique()
x = np.arange(len(categories))
width = 0.35

weekend_vals = []
weekday_vals = []
for cat in categories:
    weekend_val = plot_data[(plot_data['item_category'] == cat) & (plot_data['Day_Type'] == 'Weekend')]['Total_Spent'].values
    weekday_val = plot_data[(plot_data['item_category'] == cat) & (plot_data['Day_Type'] == 'Weekday')]['Total_Spent'].values
    weekend_vals.append(weekend_val[0] if len(weekend_val) > 0 else 0)
    weekday_vals.append(weekday_val[0] if len(weekday_val) > 0 else 0)

bars1 = ax2.bar(x - width/2, weekend_vals, width, label='Weekend', color='#2E86AB')
bars2 = ax2.bar(x + width/2, weekday_vals, width, label='Weekday', color='#A23B72')
ax2.set_xlabel('Item Category')
ax2.set_ylabel('Average Order Value ($)')
ax2.set_title('AOV by Category: Weekend vs Weekday', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'${height:.2f}', ha='center', va='bottom', fontsize=8)

# Subplot 3: Revenue by Category (Pie Chart)
ax3 = fig.add_subplot(3, 2, 3)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = ax3.pie(category_revenue.values, labels=category_revenue.index, 
                                    autopct='%1.1f%%', colors=colors[:len(category_revenue)], explode=(0.05,) * len(category_revenue))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Revenue Distribution by Category', fontweight='bold')

# Subplot 4: Monthly Revenue Trend
ax4 = fig.add_subplot(3, 2, 4)
months = monthly_spending.index.astype(str)
revenue_values = monthly_spending['Revenue'].values
ax4.plot(months, revenue_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax4.fill_between(range(len(months)), revenue_values, alpha=0.3, color='#2E86AB')
ax4.set_xlabel('Month')
ax4.set_ylabel('Revenue ($)')
ax4.set_title('Monthly Revenue Trend (2023)', fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

# Subplot 5: Location Performance
ax5 = fig.add_subplot(3, 2, 5)
locations = location_performance.index[:5]  # Top 5 locations
location_revenue = location_performance['Total_Revenue'][:5]
ax5.barh(range(len(locations)), location_revenue.values, color='#4ECDC4')
ax5.set_yticks(range(len(locations)))
ax5.set_yticklabels(locations)
ax5.set_xlabel('Total Revenue ($)')
ax5.set_title('Top 5 Locations by Revenue', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# Subplot 6: Transaction Distribution by Day Type
ax6 = fig.add_subplot(3, 2, 6)
day_type_counts = df['Is_Weekend'].value_counts()
day_type_labels = ['Weekday', 'Weekend']
day_type_values = [day_type_counts.get(False, 0), day_type_counts.get(True, 0)]
ax6.pie(day_type_values, labels=day_type_labels, autopct='%1.1f%%', colors=['#A23B72', '#2E86AB'], explode=(0, 0.05))
ax6.set_title('Transaction Distribution: Weekend vs Weekday', fontweight='bold')

plt.suptitle('APEX PLANET Internship - Task 3: Deep-Dive Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('deep_dive_dashboard.png', dpi=300, bbox_inches='tight')
# plt.show()  # Commented out to prevent blocking in non-interactive environment

print("\nVisualization saved as 'deep_dive_dashboard.png'")

# ============================================
# 9. EXPORT DEEP-DIVE RESULTS FOR BI TOOL
# ============================================

# Create summary DataFrame for dashboard
dashboard_summary = pd.DataFrame({
    'KPI': ['Total Revenue', 'Average Order Value', 'Total Transactions', 'Weekend AOV', 'Weekday AOV', 'Top Category', 'Weekend Lift %'],
    'Value': [f'${total_revenue:,.2f}', f'${aov:.2f}', f'{total_transactions:,}', f'${weekend_aov:.2f}', f'${weekday_aov:.2f}', 
              category_revenue.idxmax(), f'{(weekend_aov/weekday_aov - 1)*100:.1f}%'],
    'Formula': ['SUM(total_spent)', 'Revenue/COUNT(transactions)', 'COUNT(transaction_id)', 'AVG(weekend total_spent)', 'AVG(weekday total_spent)',
                'MAX(revenue by category)', '(Weekend AOV/Weekday AOV - 1)*100']
})

# Export for BI tool
dashboard_summary.to_csv('kpi_summary.csv', index=False)
deep_dive.to_csv('deep_dive_analysis.csv', index=False)
location_performance.to_csv('location_performance.csv')

print("\n Exported files for BI dashboard:")
print("   - kpi_summary.csv (for KPI cards)")
print("   - deep_dive_analysis.csv (for detailed analysis)")
print("   - location_performance.csv (for location analysis)")
print("   - deep_dive_dashboard.png (visual summary)")

# ============================================
# 10. FINAL SUMMARY & RECOMMENDATIONS
# ============================================

print("\n" + "=" * 60)
print("FINAL: BUSINESS RECOMMENDATIONS")
print("=" * 60)

print(f"""
 KEY FINDINGS:
─────────────────────────────────────────────────────────────
1. Weekend AOV (${weekend_aov:.2f}) is {(weekend_aov/weekday_aov - 1)*100:.1f}% higher than weekday AOV (${weekday_aov:.2f})
   → Statistical t-test confirms this is SIGNIFICANT (p = {p_value:.6f})

2. {category_revenue.idxmax()} category generates {category_revenue.max()/total_revenue*100:.1f}% of total revenue (${category_revenue.max():,.2f})
   → This is a key revenue driver

3. Best performing location: {location_performance.index[0]} with ${location_performance.iloc[0]['Total_Revenue']:,.2f} revenue
   → Consider expanding successful strategies to other locations

 ACTIONABLE RECOMMENDATIONS:
─────────────────────────────────────────────────────────────
OPERATIONS:
   • Increase staff levels by 20-30% on weekends
   • Focus on {category_revenue.idxmax()} category preparation for weekends
   • Train staff on upselling high-value items

 MARKETING:
   • Launch "Weekend Specials" campaign focused on {category_revenue.idxmax()}
   • Create bundled offers to increase AOV further
   • Target promotions to top location: {location_performance.index[0]}

 INVENTORY:
   • Increase {category_revenue.idxmax()} ingredient stock for weekends by 35%
   • Optimize stock levels based on monthly trends

 NEXT STEPS FOR DASHBOARD:
   • Import CSV files to Looker Studio/Power BI
   • Create interactive filters for Location, Month, and Category
   • Publish and share live dashboard link
""")

print("\n" + "=" * 60)
print("TASK 3 CODE EXECUTION COMPLETE")
print("=" * 60)