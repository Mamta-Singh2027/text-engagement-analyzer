#!/usr/bin/env python3
"""
Demo script to showcase the enhanced engagement analysis chatbot.
This script demonstrates the key features without requiring interactive input.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis import load_metrics, load_full_dataset, create_terminal_dashboard, generate_excel_report, parse_time_query

def demo_dashboard():
    """Demo the terminal dashboard."""
    print("\n" + "="*80)
    print("🎯 TERMINAL DASHBOARD DEMO")
    print("="*80)

    try:
        metrics = load_metrics()
        df = load_full_dataset()
        dashboard = create_terminal_dashboard(df, metrics)
        print(dashboard)
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_time_analysis():
    """Demo time-based analysis."""
    print("\n" + "="*80)
    print("📅 TIME-BASED ANALYSIS DEMO")
    print("="*80)

    try:
        df = load_full_dataset()

        # Demo month analysis
        print("\n📊 May 2025 Analysis:")
        may_data = df[df['month'] == 5]
        if len(may_data) > 0:
            high_pct = (may_data['engagement_label'] == 'high').mean() * 100
            avg_rating = may_data['rating'].mean()
            print(f"   Records: {len(may_data):,}")
            print(f"   High Engagement: {high_pct:.1f}%")
            print(f"   Average Rating: {avg_rating:.2f}/5")

        # Demo year analysis
        print("\n📊 2024 Analysis:")
        year_data = df[df['year'] == 2024]
        if len(year_data) > 0:
            engagement_counts = year_data['engagement_label'].value_counts()
            print(f"   Records: {len(year_data):,}")
            for level in ["high", "medium", "low"]:
                count = engagement_counts.get(level, 0)
                pct = (count / len(year_data) * 100)
                print(f"   {level.capitalize()}: {count:,} ({pct:.1f}%)")

    except Exception as e:
        print(f"❌ Error: {e}")

def demo_excel_export():
    """Demo Excel export."""
    print("\n" + "="*80)
    print("📈 EXCEL EXPORT DEMO")
    print("="*80)

    try:
        metrics = load_metrics()
        df = load_full_dataset()
        output_path = "demo_engagement_report.xlsx"
        generate_excel_report(df, metrics, output_path)
        print(f"✅ Excel report generated: {output_path}")
        print("\n📊 Report includes:")
        print("   • Executive Dashboard with KPIs")
        print("   • Raw data (30,000+ records)")
        print("   • Detailed analysis and insights")
        print("   • Professional formatting for stakeholders")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all demos."""
    print("🤖 Enhanced Engagement Analysis Chatbot - Demo")
    print("This demo showcases the new features of the chatbot.")

    demo_dashboard()
    demo_time_analysis()
    demo_excel_export()

    print("\n" + "="*80)
    print("🎉 DEMO COMPLETE!")
    print("="*80)
    print("\nTo use the interactive chatbot:")
    print("   python analysis.py --chat")
    print("\nTry these commands:")
    print("   • dashboard")
    print("   • show me may")
    print("   • export")
    print("   • analyze channels")
    print("   • help")

if __name__ == "__main__":
    main()