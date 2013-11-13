randn('state',0)
rand('state',0)

%Generate data
a = 1;
b = -5;
m = 100;

u = 10 * rand(m,1);
y = ( rand(m,1) < exp(a*u+b)./(1+exp(a*u+b)) );
plot(u,y,'o');
axis([-1, 11, -0.1, 1.1]);

% solve problem
%
% minimize -(sum_(y_i=1) u_i)*a - b + sum log (1 + exp(a * ui +b))

U = [ones(m,1) u];
cvx_expert true
cvx_begin
    variables x(2)
    maximize(y' * U * x - sum(log_sum_exp([zeros(1,m); x'*U'])))
cvx_end

% plot results and logistic function
ind1 = find(y==1);
ind2 = find(y==0);

aml = x(2);  bml = x(1);
us = linspace(-1,11,1000)';
ps = exp(aml*us + bml)./(1+exp(aml*us+bml));

dots = plot(us, ps, '-', u(ind1), y(ind1), 'o', u(ind2), y(ind2), 'o');

axis([-1, 11, -0.1, 1.1])
xlabel('x')
ylabel('y')