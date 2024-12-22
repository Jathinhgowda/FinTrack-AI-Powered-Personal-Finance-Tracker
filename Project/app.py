from flask import Flask, render_template, request, redirect, send_file, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import pickle
import os
import csv
from calendar import monthrange
from datetime import datetime
from io import StringIO,BytesIO
from datetime import datetime
from imblearn.over_sampling import SMOTE
import re

# Create Flask app
app = Flask(__name__)  # Corrected 'name_' typo

# Configure app
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(os.getcwd(), "database", "user_data.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey')

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure database directory exists
os.makedirs(os.path.join(os.getcwd(), "database"), exist_ok=True)

# =======================
# Database Models
# =======================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    savings_goal = db.Column(db.Float, nullable=False, default=0.0)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Helper functions
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

DATA_PATH = os.path.join('ai_model', 'training_data.csv')
MODEL_PATH = os.path.join('ai_model', 'trained_model.pkl')

# =======================
# Train the Model
# =======================

# Specify file paths
DATA_FILE = os.path.join('ai_model', 'training_data.csv')  # Update path as needed
MODEL_FILE = os.path.join('ai_model','training_data.pkl')

# Read data from the file
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}. Please ensure the file exists.")
    exit(1)

# Check if required columns exist
required_columns = [
    'Income', 'Age', 'Dependents', 'Occupation', 'City_Tier', 'Rent', 'Loan_Repayment',
    'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
    'Healthcare', 'Education', 'Miscellaneous', 'Desired_Savings_Percentage',
    'Desired_Savings', 'Disposable_Income', 'Potential_Savings_Groceries',
    'Potential_Savings_Transport', 'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment', 'Potential_Savings_Utilities',
    'Potential_Savings_Healthcare', 'Potential_Savings_Education',
    'Potential_Savings_Miscellaneous'
]
if missing_columns := [
    col for col in required_columns if col not in data.columns
]:
    print(f"Error: Missing columns in the data file: {missing_columns}")
    exit(1)

# Calculate total expenses dynamically
expense_columns = [
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out',
    'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous'
]
data['Total_Expenses'] = data[expense_columns].sum(axis=1)

# Define target variable (binary classification: positive or negative balance)
data['positive_balance'] = (data['Disposable_Income'] > 0).astype(int)

# Prepare the dataset for training
from sklearn.linear_model import LogisticRegression

# Include savings as an input feature
X = data[['Income', 'Age', 'Dependents', 'Total_Expenses', 'Disposable_Income', 'Desired_Savings']]  # Added Desired_Savings
y = data['positive_balance']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open(MODEL_FILE, 'rb') as file:
    ai_model = pickle.load(file)



def prepare_user_data(transactions):
    """Compute features dynamically from user transactions."""
    total_income = sum(t.amount for t in transactions if t.category == 'Income')
    total_expense = sum(t.amount for t in transactions if t.category == 'Expense')
    savings = total_income - total_expense  # Savings calculation
    return [total_income, total_expense, savings]

def predict_positive_balance(income, expense, savings, desired_savings):
    """Predict if the balance will be positive."""
    try:
        # Prepare input data
        input_data = [[income, income - expense, savings, expense, desired_savings]] # Added Desired_Savings
        prediction = ai_model.predict(input_data)
        return prediction[0]  # Returns 1 (positive) or 0 (negative)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None



def calculate_category_trends(user_transactions):
    """Calculate total expenses by category."""
    category_totals = {}
    for t in user_transactions:
        if t.category == 'Expense':
            category_totals[t.description] = category_totals.get(t.description, 0) + t.amount
    return sorted(category_totals.items(), key=lambda x: x[1], reverse=True)


def calculate_budget_advice(user_transactions, savings_goal):
    """Provide budgeting advice based on spending trends and savings goals."""
    category_totals = {}
    for t in user_transactions:
        if t.category == 'Expense':
            category_totals[t.description] = category_totals.get(t.description, 0) + t.amount

    total_expenses = sum(category_totals.values())

    # Budget advice for reducing large expense categories
    expense_advice = [
        f"Consider reducing spending on {category}, which accounts for {round(amount / total_expenses * 100, 2)}% of your expenses."
        for category, amount in category_totals.items()
        if amount > total_expenses * 0.2
    ]

    total_savings = sum(t.amount for t in user_transactions if t.category.lower() == 'savings')
    # Calculate how much is left to reach the savings goal
    remaining_savings = savings_goal - total_savings
    # If no savings goal is set
    if not savings_goal:
        return "Start your savings journey now! Setting a goal today will help you take control of your finances."
    
    # If the user has saved less than the goal
    if total_savings < savings_goal:
        # Calculate daily savings needed
        daily_savings_needed = remaining_savings / 10  # Assuming ₹10 savings per day
        days_needed = remaining_savings / daily_savings_needed
        
        # Advice based on days needed
        if days_needed > 100: 
            months_needed = int(days_needed / 30)  # Break it down into months
            savings_advice = f"To reach your savings goal of ₹{savings_goal}, try saving ₹{daily_savings_needed:.2f} per day. With consistent effort, you could achieve this in about {months_needed} months."
        else:
            savings_advice = f"To reach your savings goal of ₹{savings_goal}, save ₹{daily_savings_needed:.2f} per day for about {int(days_needed)} days."
    
    # If the user has already saved enough or is close
    else:
        savings_advice = f"Congratulations! You've reached your savings goal of ₹{savings_goal}. Keep it up!"

    return expense_advice + [savings_advice]


def calculate_month_analysis(user_transactions):
    """Perform month-over-month analysis of income, expenses, and savings."""
    monthly_data = {}
    for t in user_transactions:
        month = t.date.strftime('%Y-%m')
        if month not in monthly_data:
            monthly_data[month] = {'income': 0, 'expense': 0, 'savings': 0}  # Initialize 'savings'
        if t.category == 'Income':
            monthly_data[month]['income'] += t.amount
        elif t.category == 'Expense':
            monthly_data[month]['expense'] += t.amount
        elif t.category.lower() == 'savings':  # Case-insensitive match for 'savings'
            monthly_data[month]['savings'] += t.amount

    return dict(sorted(monthly_data.items()))



# =======================
# Routes
# =======================
@app.route('/')
@login_required
def index():
    # Fetch user transactions
    user_transactions = Transaction.query.filter_by(user_id=current_user.id).all()

    # Calculate financials
    total_income = sum(t.amount for t in user_transactions if t.category == 'Income')
    total_expense = sum(t.amount for t in user_transactions if t.category == 'Expense')
    balance = total_income - total_expense

    # Calculate total savings
    total_savings = sum(t.amount for t in user_transactions if t.category == 'savings')

    # Calculate balance
    balance = total_income - total_expense

    # Calculate savings goal progress
    savings_goal = current_user.savings_goal
    if savings_goal > 0:
        savings_progress = round(min((total_savings / savings_goal) * 100, 100), 2)  # Cap at 100% and round to 2 decimal points
        savings_status = f"{savings_progress:.2f}% of your goal reached!"
    else:
        savings_progress = 0
        savings_status = "No savings goal set."

    # Make a prediction
    # Calculate prediction
    if total_income and total_expense:
        prediction = predict_positive_balance(total_income, total_expense, total_savings, current_user.savings_goal)
        prediction_message = (
            "You are likely to maintain a positive balance." if prediction == 1 else
            "You may face a negative balance. Consider revising your expenses."
        )
    else:
        prediction_message = "Insufficient data for prediction."

    # Insights
    category_trends = calculate_category_trends(user_transactions)
    budget_advice = calculate_budget_advice(user_transactions,savings_goal)
    month_analysis = calculate_month_analysis(user_transactions)

    # Prepare chart data
    labels = [category for category, _ in category_trends]
    values = [amount for _, amount in category_trends]

    # Render the index page
    return render_template(
        'index.html',
        username=current_user.username,
        transactions=user_transactions,
        balance=balance if total_income and total_expense else 0,
        total_income=total_income or 0,
        total_expense=total_expense or 0,
        savings_goal=savings_goal,
        savings_progress=savings_progress,
        savings_status=savings_status,
        budget_advice=budget_advice or [],
        month_analysis=month_analysis or {},
        chart_labels=labels or [],
        chart_values=values or [],
        prediction_message=prediction_message
    )

@app.route('/update_goal', methods=['GET', 'POST'])
@login_required
def update_goal():
    if request.method == 'POST':
        new_goal = request.form.get('savings_goal')
        if not new_goal or not new_goal.isdigit() or float(new_goal) <= 0:
            flash('Invalid savings goal! Must be a positive number.', 'error')
        else:
            current_user.savings_goal = float(new_goal)
            db.session.commit()
            flash('Savings goal updated successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('update_goal.html', username=current_user.username,savings_goal=current_user.savings_goal)




@app.route('/transactions', methods=['GET', 'POST'])
@login_required
def transactions():
    if request.method == 'POST':
        description = request.form.get('description')
        amount = request.form.get('amount')
        category = request.form.get('category')
        date = request.form.get('date')

        # Input validation
        if not all([description, amount, category, date]):
            flash('All fields are required!', 'error')
            return redirect(url_for('transactions'))

        try:
            amount = float(amount)  # Convert amount to float

            # Convert date to datetime using the correct format (yyyy-mm-dd)
            date = datetime.strptime(date, '%Y-%m-%d')

            # Create the new transaction record
            new_transaction = Transaction(
                user_id=current_user.id,
                description=description,
                amount=amount,
                category=category,
                date=date
            )

            # Add transaction to the database
            db.session.add(new_transaction)
            db.session.commit()

            # Success message
            flash('Transaction added successfully!', 'success')
        except ValueError:
            flash('Invalid amount or date format!', 'error')
        except Exception as e:
            flash(f'Error adding transaction: {str(e)}', 'error')

        return redirect(url_for('transactions'))

    # Retrieve all transactions for the current user
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()

    # Render the template with the transactions data
    return render_template('transactions.html', transactions=transactions, username=current_user.username)


@app.route('/transactions/delete/<int:transaction_id>', methods=['POST'])
@login_required
def delete_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    if transaction.user_id != current_user.id:
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('transactions'))
    db.session.delete(transaction)
    db.session.commit()
    flash('Transaction deleted successfully!', 'success')
    return redirect(url_for('transactions'))

import calendar

@app.route('/download_transactions', methods=['GET', 'POST'])
def download_transactions():  # sourcery skip: last-if-guard
    if request.method == 'POST':
        # Get month and year from form (assumed format: MM/YYYY)
        month_year = request.form['month_year']
        try:
            # Parse month and year
            month, year = map(int, month_year.split('/'))
            start_date = datetime(year, month, 1)
            end_date = datetime(year, month, 1).replace(month=month % 12 + 1, day=1) if month != 12 else datetime(year + 1, 1, 1)
        except ValueError:
            return "Invalid input format. Please use MM/YYYY."

        # Query the transactions for the given month and year
        transactions = Transaction.query.filter(
            Transaction.date >= start_date,
            Transaction.date < end_date
        ).all()

        # Prepare CSV data
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Transaction ID', 'User ID', 'Category', 'Amount', 'Date'])  # Header
        
        for transaction in transactions:
            writer.writerow([transaction.id, transaction.user_id, transaction.category, transaction.amount, transaction.date])

        # Convert StringIO to BytesIO (binary mode for sending as a file)
        output.seek(0)
        byte_data = BytesIO(output.getvalue().encode())

        # Send the CSV as a downloadable file
        return send_file(byte_data, mimetype='text/csv', as_attachment=True, download_name=f"transactions_{month:02d}_{year}.csv")
    
    # If GET request, show a form to input month/year
    return render_template('download_form.html',username=current_user.username)



@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        flash('Invalid username or password!', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name=request.form.get('name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        savings_goal = request.form.get('savings_goal')

        if not username or not email or not password or not savings_goal:
            flash('All fields are required!', 'error')
        elif not is_valid_email(email):
            flash('Invalid email address!', 'error')
        elif password != confirm_password:
            flash('Passwords do not match!', 'error')
        elif not savings_goal.isdigit() or int(savings_goal) <= 0:
            flash('Savings goal must be a positive number!', 'error')
        elif User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
            flash('Email or username already registered!', 'error')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(
                name=name,
                username=username,
                email=email,
                password=hashed_password,
                savings_goal=int(savings_goal)  # Store savings goal as an integer
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':  # Corrected 'main_' typo
    with app.app_context():
        db.create_all()  # Ensure tables exist
    app.run(debug=True)