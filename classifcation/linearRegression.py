from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
print(reg.coef_)

reg=linear_model.Ridge(alpha=0.5)
reg.fit([[0,0],[0,0],[1,1]],[0,0.1,1])
print(reg.coef_,reg.intercept_)

reg=linear_model.RidgeCV(alphas=[0.1,1.0,10.0])
reg.fit([[0,0],[0,0],[1,1]],[0,1.0,10.0])
print(reg.alpha_)