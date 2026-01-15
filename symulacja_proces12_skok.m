%% symulacja procesu - reakcja na skokowy sygna³ steruj¹cy

clear all;
grubosc = 1.25; set(groot,'DefaultLineLineWidth',grubosc); set(groot,'DefaultStairLineWidth',grubosc);
colors = lines; set(groot, 'defaultAxesColorOrder', colors);

%inicjalizacja
kmin = 10; kmax = 80;
u = zeros(1, kmax);
u(kmin:kmax) = 0.5;%rozbiegówka 10 kroków

kmax= length(u);
%warunki pocz¹tkowe i inicjalizacja zmiennych 
y = zeros(1, kmax);
x = zeros(1, kmax);%stan wewnêtrzny konieczny do symulacji

for k=kmin:kmax
    [y(k), x(k)] = proces12_symulator(u(k-5), u(k-6), x(k-1), x(k-2));
end

figure;
subplot(2,1,1);
stairs(u);
xlabel('k');
ylabel('u');
grid;
subplot(2,1,2);
plot(y);
xlabel('k');
ylabel('y');
grid;

