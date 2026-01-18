%% V. Regulacja predykcyjna NPL z horyzontami Ny i Nu

clc;
close all;

% --- horyzonty ---
Ny = 10;      % horyzont predykcji
Nu = 3;       % horyzont sterowania (Nu <= Ny)

% --- parametry regulatora ---
lambda = 0.2;     % kara na przyrost sterowania
umin = -1;
umax =  1;

Nsim = 400;       % długość symulacji
k0 = 7;

yzad = zeros(1, Nsim);
yzad(50:150)  = 0.4;
yzad(150:250) = -0.6;
yzad(250:end) = 0.7;

u = zeros(1, Nsim);
y = zeros(1, Nsim);

%model = @(xk) net(xk);
model = @(xk) W * tanh(w10 + w1 * xk);

function ypred = predict_nonlinear(model, xk, du_seq, Ny)
    ypred = zeros(Ny,1);
    x = xk;

    for i = 1:Ny
        if i <= length(du_seq)
            x(1) = x(1) + du_seq(i);   % sterowanie
        end
        ypred(i) = model(x);

        % aktualizacja regresora
        x = [x(1); x(2); ypred(i); x(4)];
    end
end

for k = k0:Nsim-Ny-1

    % --- aktualny regresor ---
    xk = [u(k-5); u(k-6); y(k-1); y(k-2)];

    % --- punkt pracy ---
    y0 = model(xk);

    % --- linearyzacja numeryczna ---
    du_eps = 1e-4;
    xk_eps = xk;
    xk_eps(1) = xk_eps(1) + du_eps;
    y1 = model(xk_eps);

    a = (y1 - y0)/du_eps;   % pochodna dy/du

    % --- macierz predykcji G ---
    G = zeros(Ny, Nu);
    for i = 1:Ny
        for j = 1:Nu
            if i >= j
                G(i,j) = a;
            end
        end
    end

    % --- wektor predykcji swobodnej ---
    y_free = y0 * ones(Ny,1);

    % --- wektor wartości zadanych ---
    y_ref = yzad(k+1:k+Ny).';

    % --- rozwiązanie analityczne (LS / MPC) ---
    Kmpc = (G'*G + lambda*eye(Nu)) \ G';
    du = Kmpc * (y_ref - y_free);

    % --- zastosuj tylko pierwszy przyrost ---
    u(k) = u(k-1) + du(1);

    % --- ograniczenia ---
    u(k) = min(max(u(k), umin), umax);

    % --- symulacja OBIEKTU RZECZYWISTEGO ---
    [y(k), ~] = proces12_symulator( ...
        u(k-5), u(k-6), y(k-1), y(k-2));
end


figure;
subplot(2,1,1);
plot(y,'b','LineWidth',1.3); hold on;
plot(yzad,'r--','LineWidth',1.3);
grid on;
legend('y','y_{zad}');
ylabel('y');
title(sprintf('NPL MPC | Ny=%d, Nu=%d, \\lambda=%.2f',Ny,Nu,lambda));

subplot(2,1,2);
stairs(u,'k','LineWidth',1.3);
grid on;
ylabel('u');
xlabel('k');
title('Sterowanie');




