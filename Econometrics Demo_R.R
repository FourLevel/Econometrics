# 載入必要的套件
library(readr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(car)
library(lmtest)
library(plm)
library(broom)
library(tidyr)

# 讀取 CSV 檔案
file_path <- '銀行資料_丟迴歸分析_20240523.csv'
data <- read_csv(file_path)

# 選擇變數
colnames(data) <- c("id", "company_code", "ym", "year", "establishment_year", "roa", "s",
                    "equity_to_assets_100", "roa_std", "z_score", "total_assets", "ln_total_assets",
                    "net_income", "ln_net_income", "pb_ratio", "debt_ratio", "company_age",
                    "shareholders_percentage", "managers_percentage", "crisis_period", "covid_period")

# 設定 Panel Data
df <- data %>% 
  group_by(company_code, year)

# 標準化數量變數
df <- data %>%
  mutate(across(c(total_assets, net_income, pb_ratio, debt_ratio, company_age, 
                  shareholders_percentage, managers_percentage), 
                ~ (.-mean(.))/sd(.), .names = "std_{col}"))

# 設定 y 和 X
y <- df$z_score
X <- df %>%
  select(starts_with("std_"), crisis_period, covid_period)

# 選擇 X 變數
X_model_1 <- df %>%
  select(std_total_assets, std_pb_ratio, std_debt_ratio, std_company_age, 
         std_managers_percentage, crisis_period)

# 檢查線性關係
ggplot(df, aes(x = std_total_assets, y = z_score)) +
  geom_point() +
  labs(title = "std_total_assets vs Z-score", x = "std_total_assets", y = "Z-score")

# 繪製每個標準化變數與 z_score 的散佈圖
df_long <- df %>%
  pivot_longer(cols = starts_with("std_"), names_to = "variable", values_to = "value")

ggplot(df_long, aes(x = value, y = z_score)) +
  geom_point(color = "blue", shape = 16, alpha = 0.6) +  # 調整點的顏色、形狀和透明度
  facet_wrap(~ variable, scales = "free_x") +
  labs(title = "Standardized Variables vs Z-score", x = "Standardized Variables", y = "Z-score") +
  theme_minimal() +  # 使用簡約主題
  theme(
    plot.title = element_text(size = 14, face = "bold"),  # 調整標題字體大小和樣式
    axis.title = element_text(size = 12),  # 調整軸標題字體大小
    strip.text = element_text(size = 10)  # 調整分面標題字體大小
  )

# 檢查多重共線性
cor_matrix <- cor(X_model_1)
print(cor_matrix)

# 計算 VIF
vif_values <- vif(lm(z_score ~ ., data = X_model_1))
print(vif_values)

# 檢查自相關性
model <- lm(z_score ~ ., data = X_model_1)
dw_test <- dwtest(model)
print(dw_test)

# 執行 Breusch-Pagan 測試
bp_test <- bptest(model)
print(bp_test)

# 檢查常態分佈
ks_test <- ks.test(residuals(model), "pnorm", mean = mean(residuals(model)), sd = sd(residuals(model)))
print(ks_test)

# QQ Plot
qqnorm(residuals(model))
qqline(residuals(model))

# 殘差分佈
hist(residuals(model), breaks = 30, main = "Residuals Distribution", xlab = "Residuals", ylab = "Frequency")

# 執行 Hausman 測試
fixed_effects_model <- plm(z_score ~ std_total_assets + std_pb_ratio + std_debt_ratio + std_company_age + std_managers_percentage + crisis_period, data = df, model = "within")
random_effects_model <- plm(z_score ~ std_total_assets + std_pb_ratio + std_debt_ratio + std_company_age + std_managers_percentage + crisis_period, data = df, model = "random")
hausman_test <- phtest(fixed_effects_model, random_effects_model)
print(hausman_test)

# 清理結果
results <- tidy(fixed_effects_model)
results <- results %>%
  mutate(significance = case_when(
    p.value < 0.001 ~ "***",
    p.value < 0.01 ~ "**",
    p.value < 0.05 ~ "*",
    p.value < 0.1 ~ ".",
    TRUE ~ ""
  ))

print(results)