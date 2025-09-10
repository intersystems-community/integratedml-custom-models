#!/usr/bin/env python3
"""
Generate substantial ML datasets for IntegratedML training
Creates realistic data with proper distributions and patterns
"""

import intersystems_iris.dbapi._DBAPI as iris
import random
import datetime
from typing import List, Tuple
import math

def connect_iris():
    """Connect to IRIS database"""
    return iris.connect(
        hostname="localhost",
        port=1974,
        namespace="USER", 
        username="demo",
        password="demo"
    )

def clear_existing_data():
    """Clear existing data from all tables"""
    print("🗑️ Clearing existing data...")
    conn = connect_iris()
    cursor = conn.cursor()
    
    tables = ['CreditApplications', 'Transactions', 'SalesData', 'DNASequences']
    
    for table in tables:
        try:
            cursor.execute(f"DELETE FROM {table}")
            print(f"  ✅ Cleared {table}")
        except Exception as e:
            print(f"  ⚠️ {table}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()

def generate_credit_applications(count: int = 1000) -> List[Tuple]:
    """Generate realistic credit application data"""
    print(f"💳 Generating {count} credit applications...")
    
    employment_statuses = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Student', 'Retired']
    
    data = []
    for i in range(1, count + 1):
        # Create realistic correlations
        income = random.uniform(25000, 150000)
        credit_history = random.randint(0, 25)
        employment = random.choice(employment_statuses)
        
        # Higher income = lower debt ratio generally
        debt_ratio = max(0.1, min(0.8, random.gauss(0.3, 0.15) * (100000 / income)))
        
        # Loan amount correlated with income
        loan_amount = random.uniform(income * 0.1, income * 2.5)
        
        # Risk score based on realistic factors
        risk_base = 300
        income_factor = min(350, income / 500)  # Max 350 points for income
        history_factor = min(100, credit_history * 4)  # Max 100 points for history
        debt_penalty = debt_ratio * 200  # Penalty for high debt ratio
        employment_bonus = {'Full-time': 50, 'Part-time': 20, 'Self-employed': 10, 
                           'Unemployed': -50, 'Student': 0, 'Retired': 30}.get(employment, 0)
        
        risk_score = int(risk_base + income_factor + history_factor - debt_penalty + employment_bonus + random.uniform(-50, 50))
        risk_score = max(300, min(850, risk_score))  # Credit score range
        
        data.append((
            i, income, credit_history, round(debt_ratio, 3), 
            employment, round(loan_amount, 2), risk_score
        ))
    
    return data

def generate_transactions(count: int = 2000) -> List[Tuple]:
    """Generate realistic transaction data"""
    print(f"💰 Generating {count} transactions...")
    
    merchant_categories = ['Grocery', 'Gas', 'Restaurant', 'Retail', 'Online', 'ATM', 'Healthcare', 'Entertainment']
    
    data = []
    for i in range(1, count + 1):
        amount = random.lognormvariate(3, 1.5)  # Log-normal distribution for amounts
        amount = max(1.0, min(5000.0, amount))  # Cap at reasonable range
        
        merchant = random.choice(merchant_categories)
        hour = random.randint(0, 23)
        day_of_week = random.randint(1, 7)
        user_age = random.randint(18, 80)
        
        # Fraud probability based on realistic patterns
        fraud_prob = 0.02  # Base 2% fraud rate
        
        # Higher amounts more likely to be fraud
        if amount > 1000:
            fraud_prob *= 3
        elif amount > 500:
            fraud_prob *= 1.5
            
        # Unusual hours more likely to be fraud
        if hour < 6 or hour > 22:
            fraud_prob *= 2
            
        # ATM category more likely to be fraud
        if merchant == 'ATM':
            fraud_prob *= 2
            
        # Older users less likely to have fraud
        if user_age > 60:
            fraud_prob *= 0.5
            
        is_fraud = 1 if random.random() < min(0.15, fraud_prob) else 0
        
        data.append((
            i, round(amount, 2), merchant, hour, 
            day_of_week, user_age, is_fraud
        ))
    
    return data

def generate_sales_data(count: int = 1500) -> List[Tuple]:
    """Generate realistic sales data with seasonality"""
    print(f"📊 Generating {count} sales records...")
    
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Automotive']
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Start date
    start_date = datetime.date(2023, 1, 1)
    
    data = []
    for i in range(1, count + 1):
        # Random date in the past 2 years
        days_offset = random.randint(0, 730)
        sale_date = start_date + datetime.timedelta(days=days_offset)
        
        # Determine quarter
        quarter = f"Q{(sale_date.month - 1) // 3 + 1}"
        
        category = random.choice(categories)
        quantity = random.randint(1, 50)
        
        # Category affects base price
        base_prices = {
            'Electronics': 200, 'Clothing': 50, 'Books': 15, 
            'Home': 80, 'Sports': 60, 'Beauty': 30, 'Automotive': 150
        }
        
        base_price = base_prices[category]
        unit_price = random.uniform(base_price * 0.5, base_price * 3)
        amount = quantity * unit_price
        
        # Seasonal adjustments
        seasonal_multipliers = {
            'Q1': 0.9, 'Q2': 1.0, 'Q3': 1.1, 'Q4': 1.3  # Holiday boost in Q4
        }
        amount *= seasonal_multipliers[quarter]
        
        # Electronics boost in Q4 (holiday season)
        if category == 'Electronics' and quarter == 'Q4':
            amount *= 1.2
            
        data.append((
            i, sale_date.isoformat(), round(amount, 2), 
            quantity, category, quarter
        ))
    
    return data

def generate_dna_sequences(count: int = 800) -> List[Tuple]:
    """Generate realistic DNA sequence data"""
    print(f"🧬 Generating {count} DNA sequences...")
    
    bases = ['A', 'T', 'C', 'G']
    species = ['Human', 'Mouse', 'Rat', 'Fruit Fly', 'C. elegans']
    chromosomes = [f'Chromosome {i}' for i in range(1, 23)] + ['X', 'Y']
    
    # Species-specific patterns (simplified)
    species_patterns = {
        'Human': {'A': 0.30, 'T': 0.30, 'C': 0.20, 'G': 0.20},
        'Mouse': {'A': 0.28, 'T': 0.28, 'C': 0.22, 'G': 0.22},
        'Rat': {'A': 0.29, 'T': 0.29, 'C': 0.21, 'G': 0.21},
        'Fruit Fly': {'A': 0.31, 'T': 0.31, 'C': 0.19, 'G': 0.19},
        'C. elegans': {'A': 0.32, 'T': 0.32, 'C': 0.18, 'G': 0.18}
    }
    
    data = []
    for i in range(1, count + 1):
        species_name = random.choice(species)
        location = random.choice(chromosomes) if species_name in ['Human', 'Mouse'] else f'Scaffold {random.randint(1, 100)}'
        
        # Generate sequence with species-specific bias
        sequence_length = random.randint(50, 200)
        pattern = species_patterns[species_name]
        
        sequence = ''
        for _ in range(sequence_length):
            rand = random.random()
            if rand < pattern['A']:
                sequence += 'A'
            elif rand < pattern['A'] + pattern['T']:
                sequence += 'T'
            elif rand < pattern['A'] + pattern['T'] + pattern['C']:
                sequence += 'C'
            else:
                sequence += 'G'
        
        data.append((i, sequence, species_name, location))
    
    return data

def insert_data_batch(table_name: str, data: List[Tuple], columns: str):
    """Insert data in batches for efficiency"""
    print(f"  📥 Inserting {len(data)} records into {table_name}...")
    
    conn = connect_iris()
    cursor = conn.cursor()
    
    try:
        # Prepare statement with proper number of placeholders
        placeholders = ','.join(['?' for _ in range(len(data[0]))])
        sql = f"INSERT INTO {table_name} {columns} VALUES ({placeholders})"
        
        # Insert in batches of 100
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany(sql, batch)
            
            if i % 500 == 0:  # Progress update every 500 records
                print(f"    ⏳ Inserted {i + len(batch)}/{len(data)} records...")
        
        conn.commit()
        print(f"  ✅ Successfully inserted {len(data)} records into {table_name}")
        
    except Exception as e:
        print(f"  ❌ Failed to insert into {table_name}: {e}")
        conn.rollback()
    
    cursor.close()
    conn.close()

def verify_data():
    """Verify the generated data"""
    print("\n📋 Verifying generated data...")
    
    conn = connect_iris()
    cursor = conn.cursor()
    
    tables = [
        ('CreditApplications', 'id, income, credit_history_length, debt_to_income_ratio, employment_status, loan_amount, risk_score'),
        ('Transactions', 'id, amount, merchant_category, hour_of_day, day_of_week, user_age, is_fraud'),
        ('SalesData', 'id, sale_date, amount, quantity, product_category, quarter'),
        ('DNASequences', 'id, sequence, species, location')
    ]
    
    for table_name, _ in tables:
        try:
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Sample records
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            samples = cursor.fetchall()
            
            print(f"  📊 {table_name}: {count} records")
            for sample in samples:
                print(f"    📝 Sample: {sample}")
                
        except Exception as e:
            print(f"  ❌ {table_name}: {e}")
    
    cursor.close()
    conn.close()

def main():
    """Main execution function"""
    print("🚀 Starting ML dataset generation...")
    print("This will create realistic datasets for all 4 IntegratedML models\n")
    
    # Clear existing data
    clear_existing_data()
    
    # Generate and insert data for each model
    datasets = [
        ('CreditApplications', generate_credit_applications(1000), 
         '(id, income, credit_history_length, debt_to_income_ratio, employment_status, loan_amount, risk_score)'),
        ('Transactions', generate_transactions(2000),
         '(id, amount, merchant_category, hour_of_day, day_of_week, user_age, is_fraud)'),
        ('SalesData', generate_sales_data(1500),
         '(id, sale_date, amount, quantity, product_category, quarter)'),
        ('DNASequences', generate_dna_sequences(800),
         '(id, sequence, species, location)')
    ]
    
    for table_name, data, columns in datasets:
        insert_data_batch(table_name, data, columns)
    
    # Verify the data
    verify_data()
    
    print("\n🎉 ML dataset generation completed!")
    print("Ready for IntegratedML model training with substantial datasets")

if __name__ == "__main__":
    main()