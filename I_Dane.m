%% I. Dane
clear all;
close all;
clc;

grubosc = 1.25; set(groot,'DefaultLineLineWidth',grubosc); set(groot,'DefaultStairLineWidth',grubosc);
colors = lines; set(groot, 'defaultAxesColorOrder', colors);

u_min = -1;
u_max =  1;
Nu = 100;
u_vals = linspace(u_min,u_max,Nu);

k_min = 10;
k_max = 70;

y_stat = zeros(1,Nu);

for i = 1:Nu
    u = u_vals(i) * ones(1,k_max);
    x = zeros(1,k_max);
    y = zeros(1,k_max);

    for k = k_min:k_max
        [y(k), x(k)] = proces12_symulator(u(k-5), u(k-6), x(k-1), x(k-2));
    end

    y_stat(i) = y(k_max);
end

figure;
plot(u_vals, y_stat);
xlabel('u');
ylabel('y');
title('Charakterystyka statyczna procesu');
grid on;

%----------Dane uczące----------
rng(42)

N = 2200;
T_change = 30;      

nBlocks = ceil(N/T_change);
u_levels = u_min + (u_max-u_min)*rand(1, nBlocks);
u_ucz = repelem(u_levels, T_change);
u_ucz = u_ucz(1:N);

y_ucz = zeros(1,N);
x = zeros(1,N);

for k = 7:N
    [y_ucz(k), x(k)] = proces12_symulator( ...
        u_ucz(k-5), u_ucz(k-6), x(k-1), x(k-2));
end


%--------Dane weryfikujace--------

rng(41)

nBlocks = ceil(N/T_change);
u_levels = u_min + (u_max-u_min)*rand(1, nBlocks);
u_wer = repelem(u_levels, T_change);
u_wer = u_wer(1:N);

y_wer = zeros(1,N);
x = zeros(1,N);

for k = 7:N
    [y_wer(k), x(k)] = proces12_symulator( ...
        u_wer(k-5), u_wer(k-6), x(k-1), x(k-2));
end


k = 1:length(u_wer);

figure;
subplot(2,1,1);
stairs(k, u_ucz);
ylabel('u(k)');
title('Dane uczące');
grid on;

subplot(2,1,2);
plot(k, y_ucz);
ylabel('y(k)');
xlabel('k');
grid on;

figure;
subplot(2,1,1);
stairs(k, u_wer);
ylabel('u(k)');
title('Dane weryfikujące');
grid on;

subplot(2,1,2);
plot(k, y_wer);
ylabel('y(k)');
xlabel('k');
grid on;
