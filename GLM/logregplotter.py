import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

dF = pd.read_csv('data//csvs/dataframeV2.csv', index_col=0)
x = dF[['danceability','energy']]
y = dF[['label']].values
#ly = LabelEncoder()
#y = ly.fit_transform(y)
y = np.where(y=='metal',1,0)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

logreg = LogisticRegression(solver = 'newton-cg',multi_class='auto',max_iter=100000,n_jobs=-1)
logreg.fit(x_train,y_train)


probs=logreg.predict_proba(x_test)
mesh_size = 0.05
# How much to extend beyond min and max values (optional)
margin = 0.02

x = dF[['danceability','energy']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Identify min and max values for input variables
x_min, x_max = x_test['danceability'].min() - margin, x_test['danceability'].max() + margin
y_min, y_max = x_test['energy'].min() - margin, x_test['energy'].max() + margin

# Return evenly spaced values based on a range between min and max
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)

# Create a meshgrid
xx, yy = np.meshgrid(xrange, yrange)

# Use models to create a prediciton plane - Logistic Regression
pred_LR = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])
pred_LR = pred_LR[:,1].reshape(xx.shape)
mask_0 = y_test < 0.5   
mask_1 = y_test > 0.5

# Create a 3D scatter plot with predictions
fig = px.scatter_3d(dF, x=x_train['danceability'], y=x_train['energy'], z=y_train.ravel(), 
                 opacity=0.8, color_discrete_sequence=['black'],
                 labels=dict(x="Danceability", 
                             y="Energy",
                             z="Predicted Probability to be Metal",))

# Set figure title and colors
fig.update_layout(title_text="Scatter 3D Plot with LR Prediction Surface",
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'
                                          ),
                               zaxis=dict(backgroundcolor='white',
                                          color='black', 
                                          gridcolor='lightgrey')))
# Update marker size
fig.update_traces(marker=dict(size=3))

# Add prediction plane
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_LR, name='LR Prediction',
                          colorscale=px.colors.sequential.Sunsetdark))

#fig.add_traces(go.Scatter3d(x=x_test['danceability'][mask_1], y=x_test['energy'][mask_1], z=probs[mask_1], 
#                            name='White won', showlegend=False, mode = 'markers', opacity=0.9, marker=dict(color='limegreen', size=3)))
#fig.add_traces(go.Scatter3d(x=x_test['danceability'][mask_0], y=x_test['energy'][mask_0], z=probs[mask_0], 
#                            name='White did not win', showlegend=False, mode = 'markers', opacity=0.9, marker=dict(color='blue', size=3)))

fig.show()