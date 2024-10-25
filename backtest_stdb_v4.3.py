import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

st.set_page_config(layout="wide", page_title="Trading Strategy Backtest Dashboard")
st.title("Trading Strategy Backtest Dashboard")

# File path - Update this to your actual file path
file_path = r"C:\Users\User\Desktop\pyton\MSTR_2019_to_Present_(10-24-2024).xlsx"

def load_and_prepare_data(file_path):
    """Load and prepare the data from Excel file"""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        data = pd.read_excel(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['9ema'] = data['close'].ewm(span=9, adjust=False).mean()
        data['is_red'] = data['open'] > data['close']
        # Calculate average price for execution
        data['execution_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_date_range():
    """Get date range selection from sidebar"""
    today = datetime.now()
    periods = {
        "Day": today - timedelta(days=1),
        "Week": today - timedelta(weeks=1),
        "Month": today - timedelta(days=30),
        "Year to Date": datetime(today.year, 1, 1),
        "Year (Trailing 12 Months)": today - timedelta(days=365),
        "2 Years": today - timedelta(days=730),
        "3 Years": today - timedelta(days=1095),
        "All Available Data": None,
        "Custom": "custom"
    }
    
    selected_period = st.sidebar.selectbox("Select Time Period", list(periods.keys()))
    
    if selected_period == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", today - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", today)
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
    elif selected_period == "All Available Data":
        return None, None
    else:
        end_date = datetime.combine(today.date(), datetime.max.time())
        start_date = datetime.combine(periods[selected_period].date(), datetime.min.time())
    
    return start_date, end_date

def check_volume_condition(df, current_idx):
    """Check volume conditions for trade entry"""
    if current_idx < 6:
        return False
    current_volume = df.iloc[current_idx]['volume']
    previous_6_candles = df.iloc[current_idx-6:current_idx]
    red_candles_volume = previous_6_candles[previous_6_candles['is_red']]['volume']
    return current_volume > red_candles_volume.max() if len(red_candles_volume) > 0 else True

def check_prior_6_opens(df, current_idx):
    """Check if current close is higher than previous 6 opens"""
    if current_idx < 6:
        return False
    current_close = df.iloc[current_idx]['close']
    previous_6_opens = df.iloc[current_idx-6:current_idx]['open']
    return all(current_close > prev_open for prev_open in previous_6_opens)

def backtest_strategy(df):
    """Run the trading strategy backtest"""
    positions = []
    current_shares = 0
    consecutive_red = 0
    
    for i in range(len(df)):
        if i < 6:
            positions.append(0)
            continue
            
        current_candle = df.iloc[i]
        
        if current_candle['is_red']:
            consecutive_red += 1
        else:
            consecutive_red = 0
            
        if current_shares > 0:
            if consecutive_red >= 3 or (consecutive_red >= 2 and current_candle['close'] < current_candle['9ema']):
                current_shares = 0
            elif not current_candle['is_red'] and current_shares < 300 and current_candle['close'] > current_candle['9ema']:
                current_shares += 100
        
        elif current_shares == 0:
            if (not current_candle['is_red'] and
                check_prior_6_opens(df, i) and
                current_candle['close'] > current_candle['9ema'] and
                check_volume_condition(df, i)):
                current_shares = 100
        
        positions.append(current_shares)
    
    df['position'] = positions
    return df

def calculate_trade_metrics(results):
    """Calculate detailed trade metrics"""
    trade_changes = results[results['position'] != results['position'].shift(1)].copy()
    trade_changes['trade_type'] = np.where(trade_changes['position'] > trade_changes['position'].shift(1), 'entry', 'exit')
    
    trades = []
    current_entry = None
    current_shares = 0
    
    for idx, row in trade_changes.iterrows():
        if row['trade_type'] == 'entry':
            current_entry = row
            current_shares = row['position']
        elif row['trade_type'] == 'exit' and current_entry is not None:
            pnl = (row['execution_price'] - current_entry['execution_price']) * current_shares
            hold_time = (row['timestamp'] - current_entry['timestamp']).total_seconds() / 3600  # in hours
            trades.append({
                'entry_time': current_entry['timestamp'],
                'exit_time': row['timestamp'],
                'hold_time': hold_time,
                'pnl': pnl,
                'shares': current_shares,
                'entry_price': current_entry['execution_price']  # Track entry price for gain/loss calculation
            })
    
    if not trades:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades)
    return trades_df

def calculate_ratios(returns_series, risk_free_rate=0.02):
    """Calculate Sortino and Calmar ratios"""
    excess_returns = returns_series - (risk_free_rate / 252)  # Daily risk-free rate
    
    # Sortino Ratio
    negative_returns = returns_series[returns_series < 0]
    downside_std = np.sqrt(np.mean(negative_returns**2))
    sortino_ratio = (np.mean(excess_returns) * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0
    
    # Calmar Ratio
    max_drawdown = calculate_max_drawdown(returns_series)
    calmar_ratio = (np.mean(returns_series) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return sortino_ratio, calmar_ratio

def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown"""
    cum_returns = (1 + returns_series).cumprod()
    rolling_max = cum_returns.expanding(min_periods=1).max()
    drawdowns = cum_returns / rolling_max - 1
    return drawdowns.min()

def calculate_average_gain_loss(trades_df):
    """Calculate average gain and average loss in dollar and percentage terms"""
    gains = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]

    average_gain_size = gains['pnl'].mean() if not gains.empty else 0
    average_loss_size = losses['pnl'].mean() if not losses.empty else 0
    
    average_gain_pct = (average_gain_size / gains['entry_price'].mean() * 100) if not gains.empty else 0
    average_loss_pct = (average_loss_size / losses['entry_price'].mean() * 100) if not losses.empty else 0

    return average_gain_size, average_gain_pct, average_loss_size, average_loss_pct

def create_price_chart_with_signals(data, trades_df):
    """Create an interactive price chart with entry/exit signals"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['close'],
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add entry points (green triangles)
    entries = trades_df[['entry_time', 'shares']].copy()
    entries['price'] = entries['entry_time'].map(data.set_index('timestamp')['close'])
    fig.add_trace(go.Scatter(
        x=entries['entry_time'],
        y=entries['price'],
        mode='markers',
        name='Entry',
        marker=dict(
            symbol='triangle-up',
            size=10,
            color='green'
        ),
        hovertemplate='Entry<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add exit points (red triangles)
    exits = trades_df[['exit_time', 'shares']].copy()
    exits['price'] = exits['exit_time'].map(data.set_index('timestamp')['close'])
    fig.add_trace(go.Scatter(
        x=exits['exit_time'],
        y=exits['price'],
        mode='markers',
        name='Exit',
        marker=dict(
            symbol='triangle-down',
            size=10,
            color='red'
        ),
        hovertemplate='Exit<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Price Chart with Entry/Exit Points',
        height=600,
        margin=dict(r=100),  # Add right margin for scroll area
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title='Price',
            side='left'
        ),
        hovermode='x unified'
    )
    
    return fig

def main():
    start_date, end_date = get_date_range()
    
    # Load initial data
    data = load_and_prepare_data(file_path)
    
    if not data.empty:
        # Filter data based on selected date range
        if start_date and end_date:
            data = data[(data['timestamp'] >= start_date) & 
                         (data['timestamp'] <= end_date)]
        
        # Run backtest
        results = backtest_strategy(data)
        results['returns'] = (results['execution_price'].pct_change() * 
                              results['position'].shift(1) / 100)  # Division by 100 to account for shares
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        
        trades_df = calculate_trade_metrics(results)
        
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            # Calculate additional metrics
            sortino_ratio, calmar_ratio = calculate_ratios(results['returns'])
            avg_gain_size, avg_gain_pct, avg_loss_size, avg_loss_pct = calculate_average_gain_loss(trades_df)
            
            # Display metrics in a matrix format
            st.markdown("<h2 style='font-size: 24px;'>Detailed Performance Metrics</h2>", unsafe_allow_html=True)  # Updated header
            metrics = [
                "Total Return", "Average Trade P&L", "Largest Win", "Largest Loss", 
                "Win Rate", "Average Win Hold Time", 
                "Average Gain ($)", "Average Gain (%)", 
                "Average Loss ($)", "Average Loss (%)", 
                "Sortino Ratio", "Calmar Ratio"
            ]
            values = [
                f"${trades_df['pnl'].sum():,.2f}",
                f"${trades_df['pnl'].mean():,.2f}",
                f"${winning_trades['pnl'].max():,.2f}" if not winning_trades.empty else "$0",
                f"${losing_trades['pnl'].min():,.2f}" if not losing_trades.empty else "$0",
                f"{(len(winning_trades)/len(trades_df)*100):.1f}%",
                f"{winning_trades['hold_time'].mean():.1f}h" if not winning_trades.empty else "0h",
                f"${avg_gain_size:.2f}",
                f"{avg_gain_pct:.2f}%",
                f"${avg_loss_size:.2f}",
                f"{avg_loss_pct:.2f}%",
                f"{sortino_ratio:.2f}",
                f"{calmar_ratio:.2f}"
            ]

            # Create HTML for a matrix layout
            col_html = "<div style='display:grid; grid-template-columns: repeat(12, 1fr); font-size: 20px; gap: 5px;'>"  # Increased font size
            col_html += ''.join([f"<div style='border-bottom: 1px solid grey; text-align: center;'>{metric}</div>" for metric in metrics])
            col_html += '</div><div style="display:grid; grid-template-columns: repeat(12, 1fr); font-size: 20px; gap: 5px;">'  # Increased font size
            col_html += ''.join([f"<div style='border-bottom: 1px solid grey; text-align: center;'>{value}</div>" for value in values])
            col_html += '</div>'

            st.markdown(col_html, unsafe_allow_html=True)

            # Performance visualizations
            st.subheader("Performance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Cumulative returns chart
                fig_cumulative = px.line(results, x='timestamp', y='cumulative_returns',
                                       title="Cumulative Returns")
                st.plotly_chart(fig_cumulative, use_container_width=True)
            
            with col2:
                # Win/Loss distribution
                win_loss_data = pd.DataFrame({
                    'Category': ['Wins', 'Losses'],
                    'Count': [len(winning_trades), len(losing_trades)]
                })
                fig_pie = px.pie(win_loss_data, values='Count', names='Category',
                                title='Win/Loss Distribution',
                                color='Category', 
                                color_discrete_map={'Wins': 'green', 'Losses': 'red'})
                
                # Update pie chart font sizes
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont=dict(size=15)
                )
                fig_pie.update_layout(
                    title=dict(
                        text='Win/Loss Distribution',
                        font=dict(size=20)
                    ),
                    font=dict(size=15)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Create two columns for trade history and price chart
            col1, col2 = st.columns(2)
            
            with col1:
                # Trade history table
                st.subheader("Trade History")
                st.dataframe(trades_df)
            
            with col2:
                # Create price chart with entry/exit signals
                fig_prices = create_price_chart_with_signals(data, trades_df)
                st.plotly_chart(fig_prices)

if __name__ == "__main__":
    main()