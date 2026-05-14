#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from datetime import datetime
import warnings 
warnings.filterwarnings('ignore')

#ML imports
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import(
    classification_report,confusion_matrix,
    roc_auc_score,f1_score,precision_score,recall_score,
    make_scorer,fbeta_score,cohen_kappa_score
)
from sklearn.utils.class_weight import compute_class_weight

#advanced models
import xgboost as xgb

#for saving results
import joblib

#visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl") 

print("enhanced fraud detection")
print(f"date:{datetime.now().strftime('%y-%m-%d%H:%M:%S')}")
print("=* 50")

#define business costs
class BusinessCosts:
    """define the business costs for fraud detection"""
    def _init_(self):
        #cost of missing a fraud (false negative)
        self.fraud_cost=200 #avg graud amount

         #cost of false alarm(false positive)
        self.false_positive_cost=10 #customer dissatisfaction, support time
    
        #cost to investigate each alert
        self.investigation_cost=30 #manual review cost

def calculate_total_cost(self,confusion_matrix):
    """calculate total cost based on confusion matrix"""
    tn,fp,fn,tp=confusion_matrix.ravel()
    fraud_loss=fn*self.fraud_cost #missing fraud
    false_alarm_cost=fp*self.false_positive_cost #false alarms
    investigation_cost=(tp+fp)*self.investigation_cost #all alerts
    total_cost=fraud_loss+false_alarm_cost+investigation_cost
    return{
        'fraud_loss':fraud_loss,
        'false_alarm':false_alarm_cost,
        'investigation_cost':investigation_cost,
        'total_cost':total_cost,
        'fraud_prevented':tp*self.fraud_cost,
        'net_benefit':tp*self.fraud_cost-total_cost
    }

#initialize business costs
costs=BusinessCosts()

print(" Business Cost Structure:")
print(f"Missing a fraud (FN): ${costs.fraud_cost}")
print(f". False alarm(FP):${costs.false_positive_cost}")
print(f"investigation per alert: ${costs.investigation_cost}")
print("\n this means:")
print("its 20x more expensive to miss a fraud than to have a false alarm")
print("Every alert costs money to investigate")
print("We need to balance detection rate with operational costs")



# Visualize cost implications

fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(14, 5))

#Cost comparison
errors =['False Negative\n(Miss Fraud)', 'False Positive\n(False Alarm)']
error_costs =[costs.fraud_cost, costs.false_positive_cost + costs. investigation_cost]
colors=['#e74c3c','#f39c12']
ax1.bar(errors, error_costs, color=colors)
ax1.set_title('Cost Comparison: FN vs FP', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cost ($)')
for i, cost in enumerate(error_costs):

    ax1.text(i, cost +5, f'$(cost)', ha='center', fontweight='bold')

# ROI visualization
fraud_caught =np.arange(0, 101, 10)
false_alarms =fraud_caught *5 #Assume 5 FP for each TP
fraud_prevented_value =fraud_caught*costs.fraud_cost
total_costs= false_alarms *(costs.false_positivec_cost+ costs. investigation_cost)+\
fraud_caught* costs. investigation_cost

net_benefit= fraud_prevented_value- total_costs

ax2.plot(fraud_caught, fraud_prevented_value, label='Fraud Prevented Value', color ='#27ae60', linewidth=2)
ax2.plot(fraud_caught, total_costs, label='Total Costs', color ='#e74c3c', linewidth=2)
ax2.plot(fraud_caught, net_benefit, label='Net Benefit', color='#3498db', linewidth=3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Frauds Caught')
ax2.set_ylabel('Amount ($)')
ax2.set_title('Cost-Benefit Analysis', fontsize-14, fontweight='bold')
ax2.legend()
ax2.grid(True,alpha=0.3)

plt.tight_layout()
plt.show()

# Load the dataset

df=pd.read_csv(" ../creditcard.csv")
print(" Dataset Information:")
print(f"Total transactions: (len(df):,]")
print (f"Features: {df.shape[1]}")
print (f"Fraud rate: (df['Class'].mean()*100:.3F)%")
print(f"Average transaction amount: ${df['Amount'].mean():.2f}")
print("Average fraud amount: ${df[df['Class']==1]['Amount'].mean():.2f}")

#Calculate class weights for cost-sensitive learning 
class_weights =compute_class_weight(
    'balanced',
    classes=np.unique(df['Class']),
    y=df['Class']

)

class_weight_dict= {0: class_weights[0], 1: class_weights[1]}

print(f"\n Class Weights (for balanced learning):")
print("Normal transactions (0): {class_weights[0]:.2f}")
print(f"Fraud transactions (1): {class_weights[1]:.2f}")
print("Weight ratio: {class_weights[1]/class_weights[0]:.0f}:1")

#Prepare features and split data
feature_columns = [col for col in df.columns if col not in ['Class']]
X=df[feature_columns]
y=df['Class']

#Train-test split with stratification
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Scale features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)

print(f"\n Data Split:")
print(f"Training set: {len(X_train):,} samples ({y_train.sum()} frauds)")
print (f"Test set: {len(X_test):,}samples ({y_test.sum()} frauds)")


# Create validation set for early stopping
X_train_split, X_val_split, y_train_split, y_val_split =train_test_split(
    X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
)
print (f"\n Validation Split (for early stopping):")
print (f"Training: {len(X_train_split):,}samples")
print(f"Validation: {len(X_val_split):,} samples")

#Create cost-sensitive scorer
def cost_sensitive_score(y_true, y_pred):

"""Calculate cost-sensitive score based on business costs.
Returns negative cost (lower is better for sklearn scorers).
"""

cm=confusion_matrix(y_true, y_pred)
tn, fp, fn, tp cm.ravel()


#Calculate scale_pos_weight for imbalanced data
scale_pos_weight =len(y_train(y_train== 0)) / len(y_train[y_train ==1])

print(" Training Advanced Models with Early Stopping")

print("="* 60)

print(f"Scale positive weight: {scale_pos_weight:.2f}")

# XGBoost with early stopping
print("\n Training XGBoost with early stopping...")

xgb model xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample-0.8, 15
    colsample_bytree-0.8, 16
    scale_pos weight scale_pos weight,
    random state 42,
    early_stopping_rounds=20,
    eval_metric='logloss'
)

#Fit with evaluation set
xgb_model.fit(
    X_train split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)
print (f" XGBoost stopped at iteration: {xgb_model.best_iteration}")

# Predictions
y_pred_xgb= xgb_model.predict(X_test_scaled)
y_proba_xgb= xgb_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
xgb_metrics=calculate_business_metrics(y_test, y_pred_xgb, y_proba_xgb)
models['XGBoost'] =xgb_model
results['XGBoost']=xgb_metrics

print("\n XGBoost Test Performance:")
print(f" F1-Score: {xgb metrics['fi_score']:.4f}")
print(f" ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
print(f" Total Cost: ${xgb metrics! ['total cost']:,.0f}")
print(f" ROI:{xgb_metrics['roi']:.1f}%")

# LightGBM with early stopping
print("\n Training LightGBM with early stopping...")

lgb_model= lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

# Fit with evaluation set and early stopping
lgb_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    callbacks [lgb.early_stopping(20), lgb.log_evaluation(0)]
)

print (f" LightGBM stopped at iteration: (1gb_model.best_iteration_)")

# Predictions
y_pred_lgb = lgb_model.predict(X_test_scaled)
y_proba_1gb = lgb_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
1gb_metrics = calculate_business_metrics(y_test, y_pred_1gb, y_proba_lgb)
models['LightGBM'] = 1gb_model 
results['LightGBM] = 1gb_metrics
        
print (f"\n LightGBM Test Performance:")
print(f" F1-Score: (1gb_metrics ['fl_score']:.4f)")
print(F" ROC-AUC: (1gb metrics['roc_auc']:.4f)")
print (f" Total Cost: ${lgb_metrics['total_cost']:,.0f)")
print(f"ROI:{lgb_metrics['roi']:.1f}%")

#Create comparison DataFrame
comparison_data = []

for model_name, metrics in results.items():
    row = {
        'Model': model_name,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'Fi-Score': metrics['f1_score'],
        'ROC-AUC': metrics.get('roc_auc', 0),
        'Total Cost': metrics['total_cost"],
        'Net Benefit': metrics['net_benefit'],
        'ROI (%)': metrics['roi'],
        'Alerts': metrics['tp']+ metrics['fp'],
        'Frauds Caught': metrics ['tp'],
        'False Alarms': metrics['fp']
}


comparison_data.append(row)

comparison_df = pd.DataFrame (comparison_data)
comparison_df comparison_df.sort_values('Total Cost')

print(" Comprehensive Model Comparison")
print("="100)

print(comparison_df.round(4).to_string(index=False))

# Get feature importance from tree-based models
importance_models = ['Random Forest', 'XGBoost', 'LightGBM']
feature_importance_dict = {}

for model_name in importance_models:
    if model_name in models:
        model= models [model_name]
            if hasattr(model, 'feature_importances_'):
                feature_importance_dict[model_name] =model.feature_importances_

# Create feature importance DataFrame
feature_names= feature_columns
importance_df = pd.DataFrame (feature_importance_dict, index=feature_names)

# Calculate average importance
importance_df['Average']= importance_df.mean(axis=1)
importance_df = importance_df.sort_values('Average', ascending=False) 

# Plot top 15 features
plt.figure(figsize=(12, 8))
top_features =importance_df.head(15)

# Create horizontal bar plot
y_pos= np.arange(len(top_features))
plt.barh(y_pos, top_features['Average'], color= '#3498db')
plt.yticks (y_pos, top_features.index)
plt.xlabel('Average Feature Importance')
plt.title('Top 15 Most Important Features for Fraud Detection', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# Add importance values
for i, v in enumerate (top_features['Average']):
    plt.text(v + 0.001, i, f'(v:.4f)', va='center')

plt.tight _layout()
plt.show()

print("top 10 most important features:")
print(top_features['Average'].head(10).to_string())

#Find best model based on different criteria
best_f1_model =comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
best_cost_model= comparison_df.loc[comparison_df['Total Cost'].idxmin(), 'Model']
best_roi_model =comparison_df.loc[comparison_df['ROI (%)*'].idxmax(), 'Model']

print(" Business Recommendations")
print("="*60)

print(f"\n Best Models by Criteria:")
print(f" Best F1-Score: {best_fl_model}({comparison_df[comparison_df['Model']==best_f1_model]['F1-Score'].values[0]:.4f})")
print(f" Lowest Cost: {best_cost_model}(${comparison_df[comparison_df[ 'Model']==best_cost_model]['Total Cost'].values[0]:,.0f})")
print(f" Best ROI: {best_roi_model}({comparison_df[comparison_df['Model']==best_roi_model]['ROI (%)'].values[0]:.1f}%)")

#Calculate potential savings
baseline_cost = comparison_df['Total Cost'].max()
best_cost =comparison_df['Total Cost'].min()
savings= baseline_cost-best_cost
savings_percent =(savings / baseline_cost)*100

print("\n Operational Impact:")
print(f" Alert rate: {alert_rate:.2f}% of transactions")
print (f" Daily alerts {estimated}: {best_model_alerts *365 /2:.0f}")
print(f" Precision: {comparison_df[comparison_df['Model']==best_cost_model]['Precision'].values[0]:.1%} of alerts are actual fraud")

print("\n Recommendations:")
print(f" 1. Deploy {best_cost_model} for production use")
print(" 2. Set up monitoring for model drift and performance degradation")
print(" 3. Consider A/B testing with current system")
print(" 4. Implement feedback loop for continuous improvement")
print(" 5. Review and adjust cost parameters quarterly")

#Save all models and results

save_data = {
    'models': models,
    'scaler': scaler,
    'results': results,
    'comparison': comparison_df,
    'cost_config': {
        'fraud_cost': costs.fraud_cost,
        'false_positive_cost': costs.false_positive_cost,
        'investigation_cost': costs.investigation_cost

    },
    'feature_columns': feature_columns
}

# Save to file
joblib.dump(save_data,'../enhanced_fraud_models_tutorial.joblib')
print(" Models and results saved to 'enhanced_fraud_models_tutorial.joblib'")

# Save comparison report
comparison_df.to_csv('../model_comparison_report.csv', index=False)
print(" Comparison report saved to 'model comparison_report.csv'")