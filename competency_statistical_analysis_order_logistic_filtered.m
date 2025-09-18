% 주요 특징:
% 1. 개별 레이더 차트 생성 (통일된 스케일)
% 3. 하이퍼파라미터 튜닝 및 교차검증
% 4. 고도화된 상관 기반 가중치 시스템
% 5. LogitBoost Feature Importance와 상관분석 통합
% 6. 학습된 모델 저장 및 재사용 기능
% 7. 비활성/과활성 포함 분석, 신뢰불가 데이터 제외 추가
 
clear; clc; close all;

rng(42)  
%% ========================================================================
%                          PART 1: 초기 설정 및 데이터 로딩
% =========================================================================

% 전역 설정
set(0, 'DefaultAxesFontName', 'Malgun Gothic');
set(0, 'DefaultTextFontName', 'Malgun Gothic');
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultTextFontSize', 12);
set(0, 'DefaultLineLineWidth', 2);

% 파일 경로 설정
config = struct();
config.hr_file = 'D:\project\HR데이터\데이터\역량검사 요청 정보\최근 3년 입사자_인적정보_cleaned.xlsx';
config.comp_file = 'D:\project\HR데이터\데이터\역량검사 요청 정보\23-25년 역량검사.xlsx';
config.output_dir = 'D:\project\HR데이터\결과\자가불소';
config.timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
config.model_file = fullfile(config.output_dir, 'trained_logitboost_model.mat');
config.use_saved_model = true;  % 저장된 모델 사용 여부

% 성과 순위 정의 (위장형 소화성만 제외)
config.performance_ranking = containers.Map(...
    {'자연성', '성실한 가연성', '유익한 불연성', '유능한 불연성', ...
     '게으른 가연성', '무능한 불연성', '소화성'}, ...
    [8, 7, 6, 5, 4, 3, 1]);

% 순서형 로지스틱 회귀용 레벨 정의 (1: 최하위, 5: 최상위)
config.ordinal_levels = containers.Map(...
    {'소화성', '무능한 불연성', '게으른 가연성', '유능한 불연성', ...
     '유익한 불연성', '성실한 가연성', '자연성'}, ...
    [1, 2, 3, 4, 5, 6, 7]);

config.level_names = {'소화성', '무능한 불연성', '게으른 가연성', '유능한 불연성', ...
                     '유익한 불연성', '성실한 가연성', '자연성'};

% 고성과자/저성과자 정의 (이진 분류용) - 유능한 불연성 제외
config.high_performers = {'성실한 가연성', '자연성', '유익한 불연성'};
config.low_performers = {'무능한 불연성', '소화성', '게으른 가연성'};
config.excluded_from_analysis = {'유능한 불연성'};  % 분석에서 제외
config.excluded_types = {'위장형 소화성'}; % 위장형 소화성도 제외

%% 1.1 데이터 로딩
fprintf('【STEP 1】 데이터 로딩\n');
fprintf('────────────────────────────────────────────\n');

try
    % HR 데이터 로딩
    fprintf('▶ HR 데이터 로딩 중...\n');
    hr_data = readtable(config.hr_file, 'Sheet', 1, 'VariableNamingRule', 'preserve');
    fprintf('  ✓ HR 데이터: %d명 로드 완료\n', height(hr_data));

    % 역량검사 데이터 로딩
    fprintf('▶ 역량검사 데이터 로딩 중...\n');
    comp_upper = readtable(config.comp_file, 'Sheet', '역량검사_상위항목', 'VariableNamingRule', 'preserve');
    comp_total = readtable(config.comp_file, 'Sheet', '역량검사_종합점수', 'VariableNamingRule', 'preserve');
    fprintf('  ✓ 상위항목 데이터: %d명\n', height(comp_upper));
    fprintf('  ✓ 종합점수 데이터: %d명\n', height(comp_total));

catch ME
    error('데이터 로딩 실패: %s', ME.message);
end

%% 1.1-1 신뢰가능성 필터링 추가
fprintf('\n【STEP 1-1】 신뢰가능성 필터링\n');
fprintf('────────────────────────────────────────────\n');

% 신뢰가능성 컬럼 찾기
reliability_col_idx = find(contains(comp_upper.Properties.VariableNames, '신뢰가능성'), 1);
if ~isempty(reliability_col_idx)
    fprintf('▶ 신뢰가능성 컬럼 발견: %s\n', comp_upper.Properties.VariableNames{reliability_col_idx});
    
    % 신뢰불가 데이터 제외
    reliability_data = comp_upper{:, reliability_col_idx};
    if iscell(reliability_data)
        unreliable_idx = strcmp(reliability_data, '신뢰불가');
    else
        unreliable_idx = false(height(comp_upper), 1);
    end
    
    fprintf('  신뢰불가 데이터: %d명\n', sum(unreliable_idx));
    
    % 신뢰가능한 데이터만 유지
    comp_upper = comp_upper(~unreliable_idx, :);
    fprintf('  ✓ 신뢰가능한 데이터: %d명\n', height(comp_upper));
else
    fprintf('  ⚠ 신뢰가능성 컬럼이 없습니다. 모든 데이터를 사용합니다.\n');
end

%% 1.2 인재유형 데이터 추출 및 정제
fprintf('\n【STEP 2】 인재유형 데이터 추출 및 정제\n');
fprintf('────────────────────────────────────────────\n');

% 인재유형 컬럼 찾기
talent_col_idx = find(contains(hr_data.Properties.VariableNames, {'인재유형', '인재', '유형'}), 1);
if isempty(talent_col_idx)
    error('인재유형 컬럼을 찾을 수 없습니다.');
end

talent_col_name = hr_data.Properties.VariableNames{talent_col_idx};
fprintf('▶ 인재유형 컬럼: %s\n', talent_col_name);

% 빈 값 제거
hr_clean = hr_data(~cellfun(@isempty, hr_data{:, talent_col_idx}), :);

% 위장형 소화성만 제외
excluded_mask = strcmp(hr_clean{:, talent_col_idx}, '위장형 소화성');
hr_clean = hr_clean(~excluded_mask, :);

% 인재유형 분포 분석
talent_types = hr_clean{:, talent_col_idx};
[unique_types, ~, type_indices] = unique(talent_types);
type_counts = accumarray(type_indices, 1);

fprintf('\n전체 인재유형 분포:\n');
for i = 1:length(unique_types)
    fprintf('  • %-20s: %3d명 (%5.1f%%)\n', ...
        unique_types{i}, type_counts(i), type_counts(i)/sum(type_counts)*100);
end

%% 1.3 역량 데이터 처리 - 비활성/과활성 포함 분석
fprintf('\n【STEP 3】 역량 데이터 처리 (비활성/과활성 포함)\n');
fprintf('────────────────────────────────────────────\n');

% ID 컬럼 찾기
comp_id_col = find(contains(lower(comp_upper.Properties.VariableNames), {'id', '사번'}), 1);
if isempty(comp_id_col)
    error('역량 데이터에서 ID 컬럼을 찾을 수 없습니다.');
end

% 비활성/과활성 포함하여 분석 (모든 역량 데이터 사용)
fprintf('▶ 비활성/과활성을 포함하여 모든 역량 데이터 분석\n');

% 유효한 역량 컬럼 추출 (비활성/과활성 포함)
valid_comp_cols = {};
valid_comp_indices = [];

for i = 6:width(comp_upper)
    col_name = comp_upper.Properties.VariableNames{i};

    % 모든 역량 컬럼 포함 (비활성/과활성도 포함)
    
    % 숫자 데이터인지 확인
    col_data = comp_upper{:, i};
    if isnumeric(col_data) && ~all(isnan(col_data))
        valid_data = col_data(~isnan(col_data));
        if length(valid_data) >= 5
            % 분산이 0인 경우도 처리
            data_var = var(valid_data);
            if (data_var > 0 || length(unique(valid_data)) > 1) && ...
               all(valid_data >= 0) && all(valid_data <= 100)
                valid_comp_cols{end+1} = col_name;
                valid_comp_indices(end+1) = i;
            end
        end
    end
end

fprintf('\n포함된 모든 역량 컬럼 (%d개) - 비활성/과활성 포함:\n', length(valid_comp_cols));
for i = 1:min(length(valid_comp_cols), 10)  % 처음 10개만 출력
    fprintf('  - %s\n', valid_comp_cols{i});
end
if length(valid_comp_cols) > 10
    fprintf('  ... 외 %d개 더\n', length(valid_comp_cols) - 10);
end

if isempty(valid_comp_cols)
    error('유효한 역량 컬럼을 찾을 수 없습니다. 데이터를 확인해주세요.');
end

fprintf('\n▶ 사용할 역량 항목: %d개 (비활성/과활성 포함)\n', length(valid_comp_cols));
fprintf('  유효 역량 목록:\n');
for i = 1:min(10, length(valid_comp_cols))
    fprintf('    %d. %s\n', i, valid_comp_cols{i});
end
if length(valid_comp_cols) > 10
    fprintf('    ... 외 %d개\n', length(valid_comp_cols) - 10);
end

%% 1.4 ID 매칭 및 데이터 통합
fprintf('\n【STEP 4】 데이터 매칭 및 통합\n');
fprintf('────────────────────────────────────────────\n');

% ID 표준화
hr_ids_str = arrayfun(@(x) sprintf('%.0f', x), hr_clean.ID, 'UniformOutput', false);
comp_ids_str = arrayfun(@(x) sprintf('%.0f', x), comp_upper{:, comp_id_col}, 'UniformOutput', false);

% 교집합 찾기
[matched_ids, hr_idx, comp_idx] = intersect(hr_ids_str, comp_ids_str);

fprintf('▶ 매칭 성공: %d명\n', length(matched_ids));

% 매칭된 데이터 추출
matched_hr = hr_clean(hr_idx, :);
matched_comp = comp_upper(comp_idx, valid_comp_indices);
matched_talent_types = matched_hr{:, talent_col_idx};

% 종합점수 매칭
comp_total_ids_str = arrayfun(@(x) sprintf('%.0f', x), comp_total{:, 1}, 'UniformOutput', false);
[~, ~, total_idx] = intersect(matched_ids, comp_total_ids_str);

if ~isempty(total_idx)
    total_scores = comp_total{total_idx, end};
    fprintf('▶ 종합점수 통합: %d명\n', length(total_idx));
else
    total_scores = [];
    fprintf('⚠ 종합점수 데이터 없음\n');
end
%% ========================================================================
%            PART 2: 개선된 레이더 차트 (개별 Figure, 통일 스케일)
% =========================================================================

fprintf('\n\n╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║     PART 2: 개선된 레이더 차트 (통일 스케일)             ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

%% 2.1 유형별 프로파일 계산 및 스케일 범위 설정
fprintf('【STEP 5】 유형별 프로파일 계산 및 스케일 설정\n');
fprintf('────────────────────────────────────────────\n');

unique_matched_types = unique(matched_talent_types);
n_types = length(unique_matched_types);

% 프로파일 계산
type_profiles = zeros(n_types, length(valid_comp_cols));
profile_stats = table();
profile_stats.TalentType = unique_matched_types;
profile_stats.Count = zeros(n_types, 1);
profile_stats.CompetencyMean = zeros(n_types, 1);
profile_stats.CompetencyStd = zeros(n_types, 1);
profile_stats.TotalScoreMean = zeros(n_types, 1);
profile_stats.PerformanceRank = zeros(n_types, 1);

for i = 1:n_types
    type_name = unique_matched_types{i};
    type_mask = strcmp(matched_talent_types, type_name);
    type_comp_data = matched_comp{type_mask, :};
    type_profiles(i, :) = nanmean(type_comp_data, 1);

    % 통계 정보 수집
    profile_stats.Count(i) = sum(type_mask);
    profile_stats.CompetencyMean(i) = nanmean(type_comp_data(:));
    profile_stats.CompetencyStd(i) = nanstd(type_comp_data(:));

    % 종합점수 통계
    if ~isempty(total_scores)
        type_total_scores = total_scores(type_mask);
        profile_stats.TotalScoreMean(i) = nanmean(type_total_scores);
    end

    % 성과 순위
    if config.performance_ranking.isKey(type_name)
        profile_stats.PerformanceRank(i) = config.performance_ranking(type_name);
    end
end

% 상위 12개 주요 역량 선정 (분산 기준)
comp_variance = var(table2array(matched_comp), 0, 1, 'omitnan');
[~, var_idx] = sort(comp_variance, 'descend');
top_comp_idx = var_idx(1:min(12, length(var_idx)));
top_comp_names = valid_comp_cols(top_comp_idx);

% 전체 평균 프로파일
overall_mean_profile = nanmean(table2array(matched_comp), 1);

% 통일된 스케일 범위 계산 (모든 유형의 최소/최대값)
all_profile_data = type_profiles(:, top_comp_idx);
global_min = min(all_profile_data(:)) - 5;  % 여유값 5점
global_max = max(all_profile_data(:)) + 5;  % 여유값 5점

fprintf('▶ 통일 스케일 범위: %.1f ~ %.1f\n', global_min, global_max);
fprintf('▶ 선정된 주요 역량: %d개\n', length(top_comp_idx));

%% 2.2 개별 레이더 차트 생성
fprintf('\n【STEP 6】 개별 레이더 차트 생성\n');
fprintf('────────────────────────────────────────────\n');

% 컬러맵 설정
colors = lines(n_types);

for i = 1:n_types
    % 새로운 Figure 창 생성
    fig = figure('Position', [100 + (i-1)*50, 100 + (i-1)*30, 800, 800], ...
                 'Color', 'white', ...
                 'Name', sprintf('인재유형: %s', unique_matched_types{i}));

    % 해당 유형의 프로파일 데이터
    type_profile = type_profiles(i, top_comp_idx);
    baseline = overall_mean_profile(top_comp_idx);

    % 개선된 레이더 차트 그리기 (인라인 코드)
    data = type_profile;
    baseline_data = baseline;
    labels = top_comp_names;
    title_text = unique_matched_types{i};
    color = colors(i,:);
    min_val = global_min;
    max_val = global_max;

    % 레이더 차트 생성
    n_vars = length(data);
    angles = linspace(0, 2*pi, n_vars+1);

    % 스케일 정규화
    data_norm = (data - min_val) / (max_val - min_val);
    baseline_norm = (baseline_data - min_val) / (max_val - min_val);

    % 순환을 위해 첫 번째 값을 마지막에 추가
    data_plot = [data_norm, data_norm(1)];
    baseline_plot = [baseline_norm, baseline_norm(1)];

    % 좌표 변환
    [x_data, y_data] = pol2cart(angles, data_plot);
    [x_base, y_base] = pol2cart(angles, baseline_plot);

    hold on;

    % 그리드 그리기
    grid_levels = 5;
    for j = 1:grid_levels
        r = j / grid_levels;
        [gx, gy] = pol2cart(angles, r*ones(size(angles)));
        plot(gx, gy, 'k:', 'LineWidth', 0.8, 'Color', [0.6 0.6 0.6]);

        % 그리드 레이블
        grid_value = min_val + (max_val - min_val) * r;
        text(0, r, sprintf('%.0f', grid_value), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', ...
             'FontSize', 9, 'Color', [0.4 0.4 0.4]);
    end

    % 방사선 그리기
    for j = 1:n_vars
        plot([0, cos(angles(j))], [0, sin(angles(j))], 'k:', ...
             'LineWidth', 0.8, 'Color', [0.6 0.6 0.6]);
    end

    % 기준선
    plot(x_base, y_base, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 2);

    % 데이터 플롯
    patch(x_data, y_data, color, 'FaceAlpha', 0.4, 'EdgeColor', color, 'LineWidth', 2.5);

    % 데이터 포인트
    scatter(x_data(1:end-1), y_data(1:end-1), 60, color, 'filled', ...
            'MarkerEdgeColor', 'white', 'LineWidth', 1);

    % 레이블 및 값
    label_radius = 1.25;
    for j = 1:n_vars
        [lx, ly] = pol2cart(angles(j), label_radius);

        % 차이값 계산
        diff_val = data(j) - baseline_data(j);
        diff_str = sprintf('%+.1f', diff_val);
        if diff_val > 0
            diff_color = [0, 0.5, 0];
        else
            diff_color = [0.8, 0, 0];
        end

        text(lx, ly, sprintf('%s\n%.1f\n(%s)', labels{j}, data(j), diff_str), ...
             'HorizontalAlignment', 'center', ...
             'FontSize', 10, 'FontWeight', 'bold');
    end

    % 제목
    title(title_text, 'FontSize', 16, 'FontWeight', 'bold');

    % 범례
    legend({'평균선', '해당 유형'}, 'Location', 'best', 'FontSize', 10);

    axis equal;
    axis([-1.5 1.5 -1.5 1.5]);
    axis off;
    hold off;

    % 추가 정보 표시
    if config.performance_ranking.isKey(unique_matched_types{i})
        perf_rank = config.performance_ranking(unique_matched_types{i});
        text(0.5, -0.05, sprintf('CODE: %d', perf_rank), ...
             'Units', 'normalized', ...
             'HorizontalAlignment', 'center', ...
             'FontWeight', 'bold', 'FontSize', 14);
    end

    % Figure 저장
    % saveas(fig, sprintf('radar_chart_%s_%s.png', ...
    %        strrep(unique_matched_types{i}, ' ', '_'), config.timestamp));

    fprintf('  ✓ %s 차트 생성 완료\n', unique_matched_types{i});
end

%% ========================================================================
%                    PART 3: 고도화된 상관 기반 가중치 분석
% =========================================================================

fprintf('\n\n╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║              PART 3: 고도화된 상관 기반 가중치 분석      ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

%% 3.1 성과점수 기반 상관분석
fprintf('【STEP 7】 성과점수 기반 상관분석\n');
fprintf('────────────────────────────────────────────\n');

% 각 개인의 성과점수 할당
performance_scores = zeros(length(matched_talent_types), 1);
for i = 1:length(matched_talent_types)
    type_name = matched_talent_types{i};
    if config.performance_ranking.isKey(type_name)
        performance_scores(i) = config.performance_ranking(type_name);
    end
end

% 유효한 데이터만 선택
valid_perf_idx = performance_scores > 0;
valid_performance = performance_scores(valid_perf_idx);
valid_competencies = table2array(matched_comp(valid_perf_idx, :));

fprintf('▶ 성과점수 할당 완료: %d명\n', sum(valid_perf_idx));

%% 3.2 역량별 상관계수 계산
fprintf('\n【STEP 8】 역량-성과 상관분석\n');
fprintf('────────────────────────────────────────────\n');

% 상관계수 계산
n_competencies = length(valid_comp_cols);
correlation_results = table();
correlation_results.Competency = valid_comp_cols';
correlation_results.Correlation = zeros(n_competencies, 1);
correlation_results.PValue = zeros(n_competencies, 1);
correlation_results.Significance = cell(n_competencies, 1);
correlation_results.HighPerf_Mean = zeros(n_competencies, 1);
correlation_results.LowPerf_Mean = zeros(n_competencies, 1);
correlation_results.Difference = zeros(n_competencies, 1);
correlation_results.EffectSize = zeros(n_competencies, 1);

% 성과 상위/하위 그룹 분류 (상위 25%, 하위 25%)
perf_q75 = quantile(valid_performance, 0.75);
perf_q25 = quantile(valid_performance, 0.25);
high_perf_idx = valid_performance >= perf_q75;
low_perf_idx = valid_performance <= perf_q25;

for i = 1:n_competencies
    comp_scores = valid_competencies(:, i);
    valid_idx = ~isnan(comp_scores);

    if sum(valid_idx) >= 10
        % 상관계수 계산 (Spearman)
        [r, p] = corr(comp_scores(valid_idx), valid_performance(valid_idx), 'Type', 'Spearman');
        correlation_results.Correlation(i) = r;
        correlation_results.PValue(i) = p;

        % 유의성 표시
        if p < 0.001
            correlation_results.Significance{i} = '***';
        elseif p < 0.01
            correlation_results.Significance{i} = '**';
        elseif p < 0.05
            correlation_results.Significance{i} = '*';
        else
            correlation_results.Significance{i} = '';
        end

        % 그룹별 평균
        correlation_results.HighPerf_Mean(i) = nanmean(comp_scores(high_perf_idx));
        correlation_results.LowPerf_Mean(i) = nanmean(comp_scores(low_perf_idx));
        correlation_results.Difference(i) = correlation_results.HighPerf_Mean(i) - ...
                                           correlation_results.LowPerf_Mean(i);

        % Effect Size (Cohen's d)
        high_scores = comp_scores(high_perf_idx & valid_idx);
        low_scores = comp_scores(low_perf_idx & valid_idx);
        if length(high_scores) > 1 && length(low_scores) > 1
            pooled_std = sqrt(((length(high_scores)-1)*var(high_scores) + ...
                              (length(low_scores)-1)*var(low_scores)) / ...
                              (length(high_scores) + length(low_scores) - 2));
            correlation_results.EffectSize(i) = (mean(high_scores) - mean(low_scores)) / pooled_std;
        end
    end
end

% 가중치 계산
positive_corr = max(0, correlation_results.Correlation);
weights_corr = positive_corr / (sum(positive_corr) + eps);
correlation_results.Weight = weights_corr * 100;

correlation_results = sortrows(correlation_results, 'Correlation', 'descend');

fprintf('\n상위 10개 성과 예측 역량:\n');
fprintf('%-25s | 상관계수 | p-값 | 효과크기 | 가중치(%%)\n', '역량');
fprintf('%s\n', repmat('-', 75, 1));

for i = 1:min(10, height(correlation_results))
    fprintf('%-25s | %8.4f%s | %6.4f | %8.2f | %7.2f\n', ...
        correlation_results.Competency{i}, ...
        correlation_results.Correlation(i), correlation_results.Significance{i}, ...
        correlation_results.PValue(i), ...
        correlation_results.EffectSize(i), ...
        correlation_results.Weight(i));
end

%% 3.3 상관분석 시각화
% Figure 2: 상관분석 결과
colors_vis = struct('primary', [0.2, 0.4, 0.8], 'secondary', [0.8, 0.3, 0.2], ...
               'tertiary', [0.3, 0.7, 0.4], 'gray', [0.5, 0.5, 0.5]);

fig2 = figure('Position', [100, 100, 1600, 900], 'Color', 'white');

% 상위 15개 역량의 상관계수와 가중치
subplot(2, 2, [1, 2]);
top_15 = correlation_results(1:min(15, height(correlation_results)), :);
x = 1:height(top_15);

yyaxis left
bar(x, top_15.Correlation, 'FaceColor', colors_vis.primary, 'EdgeColor', 'none');
ylabel('상관계수', 'FontSize', 12, 'FontWeight', 'bold');
ylim([-0.2, max(top_15.Correlation)*1.2]);

yyaxis right
plot(x, top_15.Weight, '-o', 'Color', colors_vis.secondary, 'LineWidth', 2, ...
     'MarkerFaceColor', colors_vis.secondary, 'MarkerSize', 8);
ylabel('가중치 (%)', 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'XTick', x, 'XTickLabel', top_15.Competency, 'XTickLabelRotation', 45);
xlabel('역량 항목', 'FontSize', 12, 'FontWeight', 'bold');
title('역량-성과 상관분석 및 가중치', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box off;

% 누적 가중치
subplot(2, 2, [3 ,4]);
cumulative_weight = cumsum(correlation_results.Weight);
plot(cumulative_weight, 'LineWidth', 2.5, 'Color', colors_vis.tertiary);
hold on;
plot([1, length(cumulative_weight)], [80, 80], '--', 'Color', colors_vis.gray, 'LineWidth', 2);
xlabel('역량 개수', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('누적 가중치 (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('누적 설명력 분석', 'FontSize', 14, 'FontWeight', 'bold');
legend('누적 가중치', '80% 기준선', 'Location', 'southeast', 'FontSize', 10);
grid on;
box off;

sgtitle('역량-성과 상관분석 종합 결과', 'FontSize', 16, 'FontWeight', 'bold');

%% 【STEP 8-1】 7개 인재유형 간 마할라노비스 거리 분석
fprintf('\n【STEP 8-1】 7개 인재유형 간 마할라노비스 거리 분석\n');
fprintf('────────────────────────────────────────────\n\n');

% 유형별 데이터 준비
unique_types = unique(matched_talent_types);
n_types = length(unique_types);

% 각 유형의 샘플 수 확인
type_counts = zeros(n_types, 1);
for i = 1:n_types
    type_counts(i) = sum(strcmp(matched_talent_types, unique_types{i}));
end

fprintf('인재유형별 샘플 수:\n');
for i = 1:n_types
    fprintf('  %s: %d명\n', unique_types{i}, type_counts(i));
end
fprintf('\n');

% 각 유형의 평균 프로파일 계산
type_means = zeros(n_types, length(valid_comp_cols));
type_covs = cell(n_types, 1);

for i = 1:n_types
    type_mask = strcmp(matched_talent_types, unique_types{i});
    type_data = table2array(matched_comp(type_mask, :));

    % 결측값 처리 (평균으로 대체)
    for j = 1:size(type_data, 2)
        missing = isnan(type_data(:, j));
        if any(missing)
            type_data(missing, j) = nanmean(type_data(:, j));
        end
    end

    type_means(i, :) = mean(type_data, 1);

    % 샘플이 3개 이상인 경우만 공분산 계산
    if type_counts(i) >= 3
        type_covs{i} = cov(type_data);
    else
        type_covs{i} = eye(size(type_data, 2));  % 단위행렬 사용
    end
end

% 그룹별 pooled 공분산 행렬 계산 (STEP 9-1과 동일한 방식)
fprintf('그룹별 공분산 행렬 계산 중...\n');

% 유효한 유형들만 사용 (샘플 수 3개 이상)
valid_types_idx = type_counts >= 3;
valid_types = unique_types(valid_types_idx);
valid_type_means = type_means(valid_types_idx, :);
valid_type_covs = type_covs(valid_types_idx);
valid_type_counts = type_counts(valid_types_idx);

if sum(valid_types_idx) < 2
    warning('유효한 유형이 2개 미만입니다. 전체 데이터 공분산을 사용합니다.');
    X_all = table2array(matched_comp);
    for j = 1:size(X_all, 2)
        missing = isnan(X_all(:, j));
        if any(missing)
            X_all(missing, j) = nanmean(X_all(:, j));
        end
    end
    pooled_cov = cov(X_all);
    pooled_cov_reg = pooled_cov + eye(size(pooled_cov)) * 1e-6;

    % 모든 유형 간 거리 계산
    distance_matrix = zeros(n_types, n_types);
    for i = 1:n_types
        for j = i+1:n_types
            diff = type_means(i, :) - type_means(j, :);
            % 백슬래시 연산자 사용 (더 안전함)
            distance_matrix(i, j) = sqrt(diff * (pooled_cov_reg \ diff'));
            distance_matrix(j, i) = distance_matrix(i, j);
        end
    end
else
    % 유효한 유형들의 pooled 공분산 계산
    total_samples = sum(valid_type_counts);
    pooled_cov = zeros(length(valid_comp_cols), length(valid_comp_cols));

    for i = 1:length(valid_types)
        weight = (valid_type_counts(i) - 1) / (total_samples - length(valid_types));
        pooled_cov = pooled_cov + weight * valid_type_covs{i};
    end

    % 정규화 추가 (특이점 방지)
    pooled_cov_reg = pooled_cov + eye(size(pooled_cov)) * 1e-6;

    % 모든 유형 간 마할라노비스 거리 계산
    distance_matrix = zeros(n_types, n_types);

    fprintf('유형 간 마할라노비스 거리 계산 중...\n');
    for i = 1:n_types
        for j = i+1:n_types
            diff = type_means(i, :) - type_means(j, :);

            try
                % Cholesky 분해 사용 (더 안전하고 빠름)
                L = chol(pooled_cov_reg, 'lower');
                v = L \ diff';
                distance_matrix(i, j) = sqrt(v' * v);
                distance_matrix(j, i) = distance_matrix(i, j);
            catch chol_error
                warning('Cholesky 분해 실패 (%s vs %s): %s. pinv 사용.', unique_types{i}, unique_types{j}, chol_error.message);
                % pinv로 대체
                try
                    distance_matrix(i, j) = sqrt(diff * pinv(pooled_cov_reg) * diff');
                    distance_matrix(j, i) = distance_matrix(i, j);
                catch
                    % 유클리드 거리로 대체
                % 유클리드 거리로 대체
                distance_matrix(i, j) = sqrt(sum(diff.^2));
                distance_matrix(j, i) = distance_matrix(i, j);
            end
        end
    end
end

fprintf('마할라노비스 거리 계산 완료\n');

% 거리 행렬 출력
fprintf('\n【마할라노비스 거리 행렬】\n');
fprintf('%-20s', '');
for i = 1:n_types
    fprintf('%8s', unique_types{i}(1:min(7,end)));
end
fprintf('\n');

for i = 1:n_types
    fprintf('%-20s', unique_types{i});
    for j = 1:n_types
        if i == j
            fprintf('%8s', '-');
        else
            fprintf('%8.2f', distance_matrix(i, j));
        end
    end
    fprintf('\n');
end
end
% 성과 순위와 거리의 관계 분석
fprintf('\n【성과 순위와 거리 관계】\n');
fprintf('────────────────────────────────────────────\n');

% 성과 순위 맵핑
performance_ranks = zeros(n_types, 1);
for i = 1:n_types
    if config.performance_ranking.isKey(unique_types{i})
        performance_ranks(i) = config.performance_ranking(unique_types{i});
    else
        performance_ranks(i) = 0;  % 순위 없음
    end
end

% 순위 차이와 거리의 상관관계
rank_diffs = [];
distances = [];
for i = 1:n_types
    for j = i+1:n_types
        if performance_ranks(i) > 0 && performance_ranks(j) > 0
            rank_diffs(end+1) = abs(performance_ranks(i) - performance_ranks(j));
            distances(end+1) = distance_matrix(i, j);
        end
    end
end

if ~isempty(rank_diffs)
    correlation = corr(rank_diffs', distances', 'Type', 'Spearman');
    fprintf('성과 순위 차이와 마할라노비스 거리의 상관계수: %.3f\n', correlation);

    if correlation > 0.5
        fprintf('→ 성과가 다를수록 역량 프로파일도 다름 (타당한 분류)\n');
    elseif correlation > 0
        fprintf('→ 약한 양의 관계 (부분적 타당성)\n');
    else
        fprintf('→ 성과와 역량 프로파일이 일치하지 않음 (재검토 필요)\n');
    end
end

% 클러스터링 가능성 분석
fprintf('\n【유형 그룹핑 분석】\n');
fprintf('────────────────────────────────────────────\n');

% 거리 기준 그룹핑 (임계값: 1.0)
threshold = 1.0;
fprintf('거리 %.1f 이하로 묶이는 그룹:\n', threshold);

visited = false(n_types, 1);
group_num = 0;

for i = 1:n_types
    if ~visited(i)
        group_num = group_num + 1;
        group_members = unique_types(i);
        visited(i) = true;

        for j = i+1:n_types
            if ~visited(j) && distance_matrix(i, j) < threshold
                group_members = [group_members; unique_types(j)];
                visited(j) = true;
            end
        end

        if length(group_members) > 1
            fprintf('\n그룹 %d:\n', group_num);
            for k = 1:length(group_members)
                fprintf('  - %s (CODE: %d)\n', group_members{k}, ...
                    config.performance_ranking(group_members{k}));
            end

            % 그룹 내 평균 거리
            if length(group_members) == 2
                idx1 = find(strcmp(unique_types, group_members{1}));
                idx2 = find(strcmp(unique_types, group_members{2}));
                avg_dist = distance_matrix(idx1, idx2);
            else
                group_dists = [];
                for m = 1:length(group_members)-1
                    for n = m+1:length(group_members)
                        idx1 = find(strcmp(unique_types, group_members{m}));
                        idx2 = find(strcmp(unique_types, group_members{n}));
                        group_dists(end+1) = distance_matrix(idx1, idx2);
                    end
                end
                avg_dist = mean(group_dists);
            end
            fprintf('  평균 거리: %.2f\n', avg_dist);
        end
    end
end

% 가장 가까운/먼 유형 쌍 찾기 (수정된 방법)
U = triu(true(n_types), 1);
distance_vector = distance_matrix(U);

% 0이 아닌 거리만 고려
valid_distances = distance_vector(distance_vector > 0);
if ~isempty(valid_distances)
    [min_dist, ~] = min(valid_distances);
    [max_dist, ~] = max(valid_distances);

    % 원래 인덱스 찾기
    [i_all, j_all] = find(U);
    min_idx = find(distance_vector == min_dist, 1);
    max_idx = find(distance_vector == max_dist, 1);

    min_i = i_all(min_idx);
    min_j = j_all(min_idx);
    max_i = i_all(max_idx);
    max_j = j_all(max_idx);
else
    % 예외 처리
    warning('유효한 거리를 찾을 수 없습니다.');
    min_dist = NaN; max_dist = NaN;
    min_i = 1; min_j = 2; max_i = 1; max_j = 2;
end

fprintf('\n【극단 케이스】\n');
fprintf('가장 유사한 유형 쌍:\n');
fprintf('  %s ↔ %s (거리: %.2f)\n', unique_types{min_i}, unique_types{min_j}, min_dist);

fprintf('\n가장 다른 유형 쌍:\n');
fprintf('  %s ↔ %s (거리: %.2f)\n', unique_types{max_i}, unique_types{max_j}, max_dist);

% 시각화: 히트맵
try
    fprintf('\n히트맵 생성 중...\n');
    fig_heatmap = figure('Position', [100, 100, 800, 700], 'Color', 'white');

    % 거리 행렬 시각화
    imagesc(distance_matrix);
    colormap('nebula');
    c = colorbar;
    c.Label.String = '마할라노비스 거리';
    c.Label.FontSize = 12;

    title('인재유형 간 마할라노비스 거리 히트맵', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('인재유형', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('인재유형', 'FontWeight', 'bold', 'FontSize', 12);

    % 축 레이블 설정 (짧은 이름 사용)
    short_names = cell(n_types, 1);
    for i = 1:n_types
        if length(unique_types{i}) > 8
            short_names{i} = unique_types{i}(1:8);
        else
            short_names{i} = unique_types{i};
        end
    end

    set(gca, 'XTick', 1:n_types, 'XTickLabel', short_names, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:n_types, 'YTickLabel', short_names);
    set(gca, 'FontSize', 10);

    % 값 표시 (대각선 제외)
    for i = 1:n_types
        for j = 1:n_types
            if i ~= j
                distance_val = distance_matrix(i, j);
                if distance_val > 0  % 유효한 거리만 표시
                    text(j, i, sprintf('%.1f', distance_val), ...
                        'HorizontalAlignment', 'center', 'Color', 'white', ...
                        'FontWeight', 'bold', 'FontSize', 9);
                end
            end
        end
    end

    % 저장
    heatmap_filename = sprintf('type_distance_heatmap_%s.png', config.timestamp);
    try
        saveas(gcf, heatmap_filename);
        fprintf('히트맵 저장 완료: %s\n', heatmap_filename);
    catch save_error
        warning('히트맵 저장 실패: %s', save_error.message);
    end

catch plot_error
    warning('히트맵 생성 실패: %s', plot_error.message);
    fprintf('→ 텍스트 결과만 제공됩니다.\n');
end

fprintf('\n✅ 7개 유형 간 마할라노비스 거리 분석 완료\n');


%% ========================================================================
%        PART 4: Cost-Sensitive Learning 기반 고성과자 예측 시스템
% =========================================================================

fprintf('\n\n╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║    PART 4: Cost-Sensitive Learning 기반 고성과자 예측     ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

%% 4.1 데이터 준비 및 클래스 불균형 해결
fprintf('【STEP 9】 데이터 준비 및 클래스 불균형 분석\n');
fprintf('────────────────────────────────────────────\n');

% 이진분류를 위한 명확한 그룹 정의
fprintf('분류 기준 재정의:\n');
fprintf('  고성과자: %s\n', strjoin(config.high_performers, ', '));
fprintf('  저성과자: %s\n', strjoin(config.low_performers, ', '));
fprintf('  분석 제외: %s\n', strjoin(config.excluded_from_analysis, ', '));

% 원본 역량 데이터 품질 확인
X_raw = table2array(matched_comp);
fprintf('\n원본 데이터 확인:\n');
fprintf('  샘플 수: %d\n', size(X_raw, 1));
fprintf('  역량 수: %d\n', size(X_raw, 2));
fprintf('  결측값 비율: %.1f%%\n', sum(isnan(X_raw(:)))/numel(X_raw)*100);

% 유능한 불연성을 제외하고 레이블 생성
y_binary = NaN(length(matched_talent_types), 1);
for i = 1:length(matched_talent_types)
    type_name = matched_talent_types{i};
    if any(strcmp(type_name, config.high_performers))
        y_binary(i) = 1;  % 고성과자
    elseif any(strcmp(type_name, config.low_performers))
        y_binary(i) = 0;  % 저성과자
    elseif any(strcmp(type_name, config.excluded_from_analysis))
        y_binary(i) = NaN;  % 분석에서 제외
    end
end

% 유능한 불연성 제외 후 데이터 확인
excluded_count = sum(strcmp(matched_talent_types, '유능한 불연성'));
fprintf('\n유능한 불연성 %d명을 분석에서 제외\n', excluded_count);

%% STEP 9-1: 마할라노비스 거리 기반 그룹 분리 타당성 검증
fprintf('\n=== STEP 9-1: 마할라노비스 거리 기반 그룹 분리 타당성 검증 ===\n');

try
    % 분석 대상 데이터 준비 (NaN 제외)
    valid_idx = ~isnan(y_binary);
    X_for_mahal = X_raw(valid_idx, :);
    y_for_mahal = y_binary(valid_idx);

    % 완전한 케이스만 사용
    complete_cases = ~any(isnan(X_for_mahal), 2);
    X_complete = X_for_mahal(complete_cases, :);
    y_complete = y_for_mahal(complete_cases);

    if sum(complete_cases) < 10
        warning('완전한 케이스가 너무 적습니다 (%d개). 마할라노비스 거리 분석을 건너뜁니다.', sum(complete_cases));
    else
        % 고성과자와 저성과자 그룹 분리
        high_perf_idx = y_complete == 1;
        low_perf_idx = y_complete == 0;

        X_high = X_complete(high_perf_idx, :);
        X_low = X_complete(low_perf_idx, :);

        fprintf('\n그룹별 샘플 수:\n');
        fprintf('  고성과자: %d명\n', size(X_high, 1));
        fprintf('  저성과자: %d명\n', size(X_low, 1));

        if size(X_high, 1) >= 3 && size(X_low, 1) >= 3
            % 전체 공분산 행렬 계산 (pooled covariance)
            n_high = size(X_high, 1);
            n_low = size(X_low, 1);

            % 각 그룹의 공분산 행렬
            cov_high = cov(X_high);
            cov_low = cov(X_low);

            % 공통 공분산 행렬 (pooled covariance)
            pooled_cov = ((n_high - 1) * cov_high + (n_low - 1) * cov_low) / (n_high + n_low - 2);

            % 그룹 중심점 계산
            mean_high = mean(X_high);
            mean_low = mean(X_low);

            % 마할라노비스 거리 계산
            mean_diff = mean_high - mean_low;

            % 마할라노비스 거리 계산 (안전한 방법)
            pooled_cov_reg = pooled_cov + eye(size(pooled_cov)) * 1e-6;
            try
                % Cholesky 분해 사용
                L = chol(pooled_cov_reg, 'lower');
                v = L \ mean_diff';
                mahal_distance_squared = v' * v;
                mahal_distance = sqrt(mahal_distance_squared);

                fprintf('\n마할라노비스 거리 분석 결과:\n');
                fprintf('  마할라노비스 거리: %.4f\n', mahal_distance);
                fprintf('  거리 제곱: %.4f\n', mahal_distance_squared);

                % 해석 기준
                fprintf('\n해석 기준:\n');
                if mahal_distance >= 3.0
                    fprintf('  ✓ 매우 우수한 그룹 분리 (D² ≥ 3.0)\n');
                    separation_quality = 'excellent';
                elseif mahal_distance >= 2.0
                    fprintf('  ✓ 우수한 그룹 분리 (D² ≥ 2.0)\n');
                    separation_quality = 'good';
                elseif mahal_distance >= 1.5
                    fprintf('  △ 보통 수준의 그룹 분리 (D² ≥ 1.5)\n');
                    separation_quality = 'moderate';
                elseif mahal_distance >= 1.0
                    fprintf('  ⚠ 약한 그룹 분리 (D² ≥ 1.0)\n');
                    separation_quality = 'weak';
                    warning('그룹 간 분리가 약합니다. 분류 모델의 성능이 제한적일 수 있습니다.');
                else
                    fprintf('  ✗ 매우 약한 그룹 분리 (D² < 1.0)\n');
                    separation_quality = 'very_weak';
                    warning('그룹 간 분리가 매우 약합니다. 분류 모델 적용을 재검토해야 합니다.');
                end

                % 효과 크기 계산 (Cohen's d equivalent for multivariate)
                effect_size = mahal_distance / sqrt(size(X_complete, 2));
                fprintf('  다변량 효과 크기: %.4f\n', effect_size);

                if effect_size >= 0.8
                    fprintf('  → 큰 효과 크기 (≥ 0.8)\n');
                elseif effect_size >= 0.5
                    fprintf('  → 중간 효과 크기 (≥ 0.5)\n');
                elseif effect_size >= 0.2
                    fprintf('  → 작은 효과 크기 (≥ 0.2)\n');
                else
                    fprintf('  → 매우 작은 효과 크기 (< 0.2)\n');
                end

                % 통계적 유의성 검정 (Hotelling's T² test)
                n_total = n_high + n_low;
                hotelling_t2 = (n_high * n_low / n_total) * mahal_distance_squared;

                % F-통계량으로 변환
                p_features = size(X_complete, 2);
                f_stat = ((n_total - p_features - 1) / ((n_total - 2) * p_features)) * hotelling_t2;
                df1 = p_features;
                df2 = n_total - p_features - 1;

                % F-분포를 이용한 p-값 계산 (근사치)
                if df2 > 0
                    p_value_approx = 1 - fcdf(f_stat, df1, df2);
                    fprintf('\n통계적 유의성 검정 (Hotelling''s T²):\n');
                    fprintf('  F-통계량: %.4f (df1=%d, df2=%d)\n', f_stat, df1, df2);
                    fprintf('  p-값 (근사): %.6f\n', p_value_approx);

                    if p_value_approx < 0.001
                        fprintf('  ✓ 매우 유의한 그룹 차이 (p < 0.001)\n');
                    elseif p_value_approx < 0.01
                        fprintf('  ✓ 유의한 그룹 차이 (p < 0.01)\n');
                    elseif p_value_approx < 0.05
                        fprintf('  ✓ 유의한 그룹 차이 (p < 0.05)\n');
                    else
                        fprintf('  ⚠ 그룹 차이가 유의하지 않음 (p ≥ 0.05)\n');
                    end
                end

                % 개별 변수별 기여도 분석
                fprintf('\n변수별 그룹 분리 기여도 (상위 5개):\n');
                try
                    % 안전한 diag 계산
                    L = chol(pooled_cov_reg, 'lower');
                    inv_diag = 1 ./ sum(L.^2, 1);  % 대각원소 근사
                    individual_contributions = abs(mean_diff .* inv_diag);
                catch
                    % 대체 방법: pinv의 대각원소
                    individual_contributions = abs(mean_diff .* diag(pinv(pooled_cov_reg))');
                end
                [sorted_contrib, sorted_idx] = sort(individual_contributions, 'descend');

                for i = 1:min(5, length(sorted_contrib))
                    var_idx = sorted_idx(i);
                    fprintf('  %s: %.4f\n', valid_comp_cols{var_idx}, sorted_contrib(i));
                end

                % 결과 저장
                mahalanobis_results = struct();
                mahalanobis_results.distance = mahal_distance;
                mahalanobis_results.distance_squared = mahal_distance_squared;
                mahalanobis_results.separation_quality = separation_quality;
                mahalanobis_results.effect_size = effect_size;
                mahalanobis_results.n_high_performers = n_high;
                mahalanobis_results.n_low_performers = n_low;
                mahalanobis_results.variable_contributions = individual_contributions;

                if exist('p_value_approx', 'var')
                    mahalanobis_results.p_value = p_value_approx;
                    mahalanobis_results.f_statistic = f_stat;
                end

                % 권장사항 출력
                fprintf('\n분석 권장사항:\n');
                if strcmp(separation_quality, 'excellent') || strcmp(separation_quality, 'good')
                    fprintf('  ✓ 그룹 분리가 우수합니다. 분류 모델 적용에 적합합니다.\n');
                elseif strcmp(separation_quality, 'moderate')
                    fprintf('  △ 보통 수준의 분리입니다. 모델 성능을 주의깊게 모니터링하세요.\n');
                else
                    fprintf('  ⚠ 그룹 분리가 약합니다. 다음을 고려하세요:\n');
                    fprintf('    - 추가적인 특성 엔지니어링\n');
                    fprintf('    - 더 정교한 분류 기준 재검토\n');
                    fprintf('    - 비선형 모델 적용 검토\n');
                end

            catch chol_error
                warning('Cholesky 분해 실패: %s. pinv 사용.', chol_error.message);
                % pinv로 대체
                try
                    mahal_distance_squared = mean_diff * pinv(pooled_cov_reg) * mean_diff';
                    mahal_distance = sqrt(mahal_distance_squared);
                catch
                    warning('모든 마할라노비스 계산 실패. 유클리드 거리 사용.');
                    mahal_distance = sqrt(sum(mean_diff.^2));
                    mahal_distance_squared = mahal_distance^2;
                end
                fprintf('  → 차원 축소나 정규화를 고려하세요.\n');
            end
        else
            warning('각 그룹에 최소 3개 이상의 샘플이 필요합니다.');
        end
    end

catch mahal_error
    warning('마할라노비스 거리 분석 중 오류 발생: %s', mahal_error.message);
    fprintf('→ 다음 단계로 진행합니다.\n');
end

fprintf('\n【2단계 결측값 처리】\n');

% Step 1: 고품질 샘플만 선택 (결측률 30% 미만)
missing_rate_per_sample = sum(isnan(X_raw), 2) / size(X_raw, 2);
quality_threshold = 0.3;
quality_samples = missing_rate_per_sample < quality_threshold;

fprintf('Step 1 - 고품질 샘플 선택:\n');
fprintf('  결측률 30%% 이상 제거: %d명 제외\n', sum(~quality_samples));
fprintf('  남은 샘플: %d명\n', sum(quality_samples));

% 유능한 불연성 제외 + 품질 필터 적용
binary_valid_idx = ~isnan(y_binary);
final_idx = binary_valid_idx & quality_samples;
X_quality = X_raw(final_idx, :);
y_quality = y_binary(final_idx);

% Step 2: 완전한 케이스만 사용 (남은 결측값 제거)
complete_cases = ~any(isnan(X_quality), 2);
X_final = X_quality(complete_cases, :);
y_final = y_quality(complete_cases);

fprintf('\nStep 2 - 완전한 케이스만 사용:\n');
fprintf('  추가 제거: %d명\n', sum(~complete_cases));
fprintf('  최종 분석 샘플: %d명\n', length(y_final));
fprintf('  - 고성과자: %d명\n', sum(y_final == 1));
fprintf('  - 저성과자: %d명\n', sum(y_final == 0));
fprintf('  - 샘플/변수 비율: %.1f\n', length(y_final)/size(X_final, 2));

% 결측값 대체 없이 완전한 데이터만 사용
X_imputed = X_final;  % 이름은 유지하되 실제로는 대체 없음
y_weight = y_final;

% 클래스 분포 확인
n_high = sum(y_weight == 1);
n_low = sum(y_weight == 0);
total_binary = length(y_weight);

fprintf('\n최종 이진분류 데이터 분포:\n');
fprintf('  고성과자 (1): %d명 (%.1f%%)\n', n_high, n_high/total_binary*100);
fprintf('  저성과자 (0): %d명 (%.1f%%)\n', n_low, n_low/total_binary*100);
fprintf('  불균형 비율: %.2f:1\n', max(n_high, n_low)/min(n_high, n_low));

% 클래스 가중치 계산 (inverse frequency weighting)
class_weights = length(y_weight) ./ (2 * [sum(y_weight==0), sum(y_weight==1)]);
sample_weights = zeros(size(y_weight));
sample_weights(y_weight==0) = class_weights(1);  % 저성과자
sample_weights(y_weight==1) = class_weights(2);  % 고성과자

fprintf('  클래스 가중치 - 저성과자: %.3f, 고성과자: %.3f\n', class_weights(1), class_weights(2));

% 비용 행렬 정의 (저성과자→고성과자 오분류 비용 1.5배)
cost_matrix = [0 1; 1.5 0];  % [TN FP; FN TP]
fprintf('  비용 행렬: 저성과자→고성과자 오분류 비용 1.5배 적용\n');

%% 4.2 모든 역량 feature 전처리 (완전한 데이터로 모든 역량 유지)
fprintf('\n【STEP 10】 모든 역량 feature 전처리\n');
fprintf('────────────────────────────────────────────\n');

% 모든 역량 feature 사용 (완전한 데이터이므로 제거할 필요 없음)
feature_names = valid_comp_cols;
n_features = size(X_imputed, 2);

fprintf('  활용 역량 feature 수: %d개 (완전한 데이터)\n', n_features);

% 표준화는 LOO-CV 내부에서 수행 (데이터 누수 방지)
fprintf('  표준화는 교차검증 내부에서 수행됩니다 (데이터 누수 방지)\n');

%% 4.3 Leave-One-Out 교차검증으로 최적 Lambda 찾기
fprintf('\n【STEP 11】 Leave-One-Out 교차검증으로 최적 Lambda 찾기\n');
fprintf('────────────────────────────────────────────\n');

% Lambda 파라미터 범위 설정
lambda_range = logspace(-3, 0, 10);  % 0.001 ~ 1.0
fprintf('Lambda 후보값: [%.4f ~ %.4f], %d개 지점\n', min(lambda_range), max(lambda_range), length(lambda_range));

% Leave-One-Out 교차검증 수행
cv_scores = zeros(length(lambda_range), 1);
cv_aucs = zeros(length(lambda_range), 1);

fprintf('\nLOO-CV 진행상황:\n');
for lambda_idx = 1:length(lambda_range)
    current_lambda = lambda_range(lambda_idx);

    % LOO-CV를 위한 예측값 저장
    loo_predictions = zeros(length(y_weight), 1);
    loo_probabilities = zeros(length(y_weight), 1);

    % Leave-One-Out 루프
    for i = 1:length(y_weight)
        % 훈련 데이터 (i번째 샘플 제외)
        train_idx = true(length(y_weight), 1);
        train_idx(i) = false;

        X_train = X_imputed(train_idx, :);  % 원본 데이터 사용
        y_train = y_weight(train_idx);
        X_test = X_imputed(i, :);  % 원본 데이터 사용

        % ★ 훈련 세트로만 표준화 (데이터 누수 방지)
        mu = mean(X_train, 1);
        sigma = std(X_train, 0, 1);
        sigma(sigma == 0) = 1;  % 0 방지

        X_train_z = (X_train - mu) ./ sigma;
        X_test_z = (X_test - mu) ./ sigma;

        % 가중치 계산 (클래스 불균형 + 비용 반영)
        w = zeros(size(y_train));
        n_class0 = sum(y_train == 0);
        n_class1 = sum(y_train == 1);

        % 역빈도 가중치 * 비용 행렬
        w(y_train == 0) = (length(y_train)/(2*n_class0)) * cost_matrix(1,2);
        w(y_train == 1) = (length(y_train)/(2*n_class1)) * cost_matrix(2,1);

        try
            mdl = fitclinear(X_train_z, y_train, ...
                'Learner', 'logistic', ...
                'Regularization', 'ridge', ...
                'Lambda', current_lambda, ...
                'Solver', 'lbfgs', ...
                'Weights', w);  % Cost 대신 Weights 사용

            [pred_label, pred_score] = predict(mdl, X_test_z);
            loo_predictions(i) = pred_label;
            loo_probabilities(i) = pred_score(2);
        catch
            % 폴백
            loo_predictions(i) = mode(y_train);
            loo_probabilities(i) = mean(y_train);
        end
    end

    % 성능 평가
    accuracy = mean(loo_predictions == y_weight);

    % AUC 계산
    try
        [~, ~, ~, auc] = perfcurve(y_weight, loo_probabilities, 1);
        cv_aucs(lambda_idx) = auc;
    catch
        cv_aucs(lambda_idx) = 0.5;  % 기본값
    end

    cv_scores(lambda_idx) = accuracy;

    fprintf('  λ=%.4f: 정확도=%.3f, AUC=%.3f\n', current_lambda, accuracy, cv_aucs(lambda_idx));
end

% 최적 Lambda 선택 (AUC 기준)
[best_auc, best_idx] = max(cv_aucs);
optimal_lambda = lambda_range(best_idx);

fprintf('\n최적 Lambda 선택:\n');
fprintf('  최적 λ: %.4f\n', optimal_lambda);
fprintf('  최고 AUC: %.3f\n', best_auc);
fprintf('  해당 정확도: %.3f\n', cv_scores(best_idx));

%% 4.4 최적 Lambda로 최종 모델 학습 및 가중치 추출
fprintf('\n【STEP 12】 최종 Cost-Sensitive 모델 학습\n');
fprintf('────────────────────────────────────────────\n');

% 전체 데이터로 표준화 (최종 모델용)
mu_final = mean(X_imputed, 1);
sigma_final = std(X_imputed, 0, 1);
sigma_final(sigma_final == 0) = 1;
X_normalized = (X_imputed - mu_final) ./ sigma_final;

% 가중치 계산
sample_weights = zeros(size(y_weight));
n0 = sum(y_weight == 0);
n1 = sum(y_weight == 1);
sample_weights(y_weight == 0) = (length(y_weight)/(2*n0)) * cost_matrix(1,2);
sample_weights(y_weight == 1) = (length(y_weight)/(2*n1)) * cost_matrix(2,1);

try
    final_mdl = fitclinear(X_normalized, y_weight, ...
        'Learner', 'logistic', ...
        'Regularization', 'ridge', ...
        'Lambda', optimal_lambda, ...
        'Solver', 'lbfgs', ...
        'Weights', sample_weights);  % Cost 제거, Weights만 사용

    coefficients = final_mdl.Beta;
    intercept = final_mdl.Bias;

    fprintf('  ✓ Cost-Sensitive 로지스틱 회귀 학습 성공\n');
    fprintf('  절편: %.4f\n', intercept);

    % 양수 계수만 사용하여 가중치 변환
    positive_coefs = max(0, coefficients);
    final_weights = positive_coefs / sum(positive_coefs) * 100;  % 백분율로 변환

    fprintf('  양수 계수 개수: %d/%d\n', sum(positive_coefs > 0), length(coefficients));
    fprintf('  가중치 변환 완료 (백분율)\n');

catch ME
    warning('모델 학습 실패: %s. 상관계수로 대체합니다.', ME.message);
    correlations = zeros(n_features, 1);
    for i = 1:n_features
        correlations(i) = corr(X_normalized(:,i), y_weight, 'rows', 'complete');
    end
    coefficients = correlations;  % ★ 중요: coefficients 변수 설정
    intercept = 0;
end

%% 4.5 모델 성능 평가 및 검증
fprintf('\n【STEP 13】 모델 성능 평가 및 검증\n');
fprintf('────────────────────────────────────────────\n');

% 종합점수 계산 (가중치 적용)
weighted_scores = X_normalized * (final_weights / 100);  % 백분율을 다시 비율로 변환

% 고성과자와 저성과자의 종합점수 비교
high_idx = y_weight == 1;
low_idx = y_weight == 0;
high_scores = weighted_scores(high_idx);
low_scores = weighted_scores(low_idx);

fprintf('종합점수 검증:\n');
fprintf('  고성과자 평균: %.3f ± %.3f (n=%d)\n', mean(high_scores), std(high_scores), length(high_scores));
fprintf('  저성과자 평균: %.3f ± %.3f (n=%d)\n', mean(low_scores), std(low_scores), length(low_scores));
fprintf('  점수 차이: %.3f\n', mean(high_scores) - mean(low_scores));

% 통계적 유의성 검정
[~, ttest_p, ~, ttest_stats] = ttest2(high_scores, low_scores);
fprintf('  t-test: t=%.3f, p=%.6f\n', ttest_stats.tstat, ttest_p);

% Effect Size (Cohen's d) 계산
pooled_std = sqrt(((length(high_scores)-1)*var(high_scores) + ...
                  (length(low_scores)-1)*var(low_scores)) / ...
                  (length(high_scores) + length(low_scores) - 2));
cohens_d = (mean(high_scores) - mean(low_scores)) / pooled_std;

fprintf('  Cohen''s d: %.3f', cohens_d);
if cohens_d < 0.2
    fprintf(' (작은 효과)\n');
elseif cohens_d < 0.5
    fprintf(' (중간 효과)\n');
elseif cohens_d < 0.8
    fprintf(' (큰 효과)\n');
else
    fprintf(' (매우 큰 효과)\n');
end

% ROC 분석
[X_roc, Y_roc, T_roc, AUC] = perfcurve(y_weight, weighted_scores, 1);
fprintf('  분류 성능 (AUC): %.3f\n', AUC);

% 최적 임계값 찾기 (Youden's J statistic)
J = Y_roc - X_roc;
[~, opt_idx] = max(J);
optimal_threshold = T_roc(opt_idx);
fprintf('  최적 임계값: %.3f (민감도=%.3f, 특이도=%.3f)\n', ...
        optimal_threshold, Y_roc(opt_idx), 1-X_roc(opt_idx));

%% 4.6 가중치 결과 분석 및 저장
fprintf('\n【STEP 14】 가중치 결과 분석 및 저장\n');
fprintf('────────────────────────────────────────────\n');

% 가중치 결과 테이블 생성
weight_results = table();
weight_results.Feature = feature_names';
weight_results.Weight_Percent = final_weights;
weight_results.Raw_Coefficient = coefficients;

% 기여도가 있는 역량만 필터링 (0.1% 이상)
significant_idx = final_weights > 0.1;
weight_results_significant = weight_results(significant_idx, :);
weight_results_significant = sortrows(weight_results_significant, 'Weight_Percent', 'descend');

fprintf('주요 역량 가중치 (기여도 0.1%% 이상):\n');
fprintf('순위 | %-25s | 가중치(%%) | 원계수\n', '역량명');
fprintf('%s\n', repmat('-', 70, 1));

for i = 1:min(15, height(weight_results_significant))
    fprintf('%2d   | %-25s | %8.2f | %8.4f\n', ...
            i, weight_results_significant.Feature{i}, ...
            weight_results_significant.Weight_Percent(i), ...
            weight_results_significant.Raw_Coefficient(i));
end

% 가중치 파일 저장
result_data = struct();
result_data.final_weights = final_weights;
result_data.feature_names = feature_names;
result_data.optimal_lambda = optimal_lambda;
result_data.optimal_threshold = optimal_threshold;
result_data.model_performance = struct('AUC', AUC, 'cohens_d', cohens_d, ...
                                      'accuracy', cv_scores(best_idx));
result_data.cost_matrix = cost_matrix;
result_data.class_weights = class_weights;

weight_filename = sprintf('cost_sensitive_weights_%s.mat', datestr(now, 'yyyy-mm-dd_HHMMSS'));
save(weight_filename, 'result_data', 'weight_results_significant');

fprintf('\n가중치 저장 완료: %s\n', weight_filename);

%% 4.7 종합 시각화
fprintf('\n【STEP 15】 Cost-Sensitive Learning 결과 시각화\n');
fprintf('────────────────────────────────────────────\n');

% 종합 시각화 생성
fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'white');

% 1. Lambda 최적화 과정
subplot(2, 3, 1);
yyaxis left
plot(lambda_range, cv_scores, 'o-', 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
ylabel('정확도', 'Color', [0.2, 0.4, 0.8]);
ylim([min(cv_scores)-0.02, max(cv_scores)+0.02]);

yyaxis right
plot(lambda_range, cv_aucs, 's-', 'LineWidth', 2, 'Color', [0.8, 0.3, 0.2]);
ylabel('AUC', 'Color', [0.8, 0.3, 0.2]);
ylim([min(cv_aucs)-0.02, max(cv_aucs)+0.02]);

% 최적점 표시
hold on;
yyaxis right;
plot(optimal_lambda, best_auc, 'r*', 'MarkerSize', 12, 'LineWidth', 3);

set(gca, 'XScale', 'log');
xlabel('Lambda (정규화 강도)');
title('LOO-CV Lambda 최적화');
grid on;

% 2. 주요 가중치 (상위 12개)
subplot(2, 3, 2);
top_n = min(12, height(weight_results_significant));
top_weights = weight_results_significant(1:top_n, :);

barh(1:top_n, top_weights.Weight_Percent, 'FaceColor', [0.3, 0.7, 0.4]);
set(gca, 'YTick', 1:top_n, 'YTickLabel', top_weights.Feature, 'FontSize', 9);
xlabel('가중치 (%)');
title('주요 역량 가중치 (상위 12개)');
grid on;

% 3. 종합점수 분포 비교
subplot(2, 3, 3);
bin_edges = linspace(min(weighted_scores), max(weighted_scores), 15);
histogram(low_scores, bin_edges, 'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.7, ...
          'DisplayName', sprintf('저성과자 (n=%d)', length(low_scores)));
hold on;
histogram(high_scores, bin_edges, 'FaceColor', [0.3, 0.7, 0.3], 'FaceAlpha', 0.7, ...
          'DisplayName', sprintf('고성과자 (n=%d)', length(high_scores)));

% 최적 임계값 표시
line([optimal_threshold, optimal_threshold], ylim, 'Color', 'k', 'LineStyle', '--', ...
     'LineWidth', 2, 'DisplayName', sprintf('최적 임계값 (%.3f)', optimal_threshold));

xlabel('종합점수');
ylabel('빈도');
title('고성과자 vs 저성과자 점수 분포');
legend('Location', 'best');
grid on;

% 4. ROC 곡선
subplot(2, 3, 4);
plot(X_roc, Y_roc, 'LineWidth', 3, 'Color', [0.2, 0.4, 0.8]);
hold on;
plot([0, 1], [0, 1], '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5);

% 최적점 표시
plot(X_roc(opt_idx), Y_roc(opt_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('위양성률 (1-특이도)');
ylabel('민감도');
title(sprintf('ROC 곡선 (AUC=%.3f)', AUC));
legend({'ROC 곡선', '무작위', '최적점'}, 'Location', 'southeast');
grid on;

% 5. 클래스별 가중치 기여도
subplot(2, 3, 5);
positive_weights = final_weights(final_weights > 0);
pie_data = [sum(positive_weights), 100 - sum(positive_weights)];
pie_labels = {sprintf('활성 역량\n(%.1f%%)', pie_data(1)), ...
              sprintf('비활성 역량\n(%.1f%%)', pie_data(2))};

pie(pie_data, pie_labels);
title('역량 활용도');
colormap([0.3, 0.7, 0.4; 0.8, 0.8, 0.8]);

% 6. 성능 지표 요약
subplot(2, 3, 6);
axis off;

% 성능 지표 텍스트
perf_text = {
    sprintf('◆ Cost-Sensitive Learning 결과 ◆');
    '';
    sprintf('최적 Lambda: %.4f', optimal_lambda);
    sprintf('교차검증 AUC: %.3f', best_auc);
    sprintf('교차검증 정확도: %.3f', cv_scores(best_idx));
    '';
    sprintf('Cohen''s d: %.3f', cohens_d);
    sprintf('최적 임계값: %.3f', optimal_threshold);
    sprintf('활성 역량 수: %d/%d', sum(final_weights > 0.1), length(final_weights));
    '';
    sprintf('클래스 가중치:');
    sprintf('  저성과자: %.3f', class_weights(1));
    sprintf('  고성과자: %.3f', class_weights(2));
    '';
    sprintf('비용 행렬: [0, 1; 1.5, 0]');
};

text(0.05, 0.95, perf_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
     'FontSize', 10, 'FontWeight', 'bold');

sgtitle('Cost-Sensitive Learning 기반 고성과자 예측 시스템 분석 결과', ...
        'FontSize', 14, 'FontWeight', 'bold');

% 그래프 저장
chart_filename = sprintf('cost_sensitive_analysis_%s.png', datestr(now, 'yyyy-mm-dd_HHMMSS'));
saveas(fig, chart_filename);

fprintf('  ✓ 시각화 차트 저장: %s\n', chart_filename);

%% 4.8 Bootstrap을 통한 가중치 안정성 검증
fprintf('\n【STEP 16】 Bootstrap 가중치 안정성 검증\n');
fprintf('────────────────────────────────────────────\n');


bootstrap_chart_filename='D:\project\HR데이터\결과\자가불소\bootstrap.xlsx';

% Bootstrap 설정
n_bootstrap = 1000;
n_samples = size(X_final, 1);
n_features = size(X_final, 2);

% Bootstrap 결과 저장 배열
bootstrap_weights = zeros(n_features, n_bootstrap);
bootstrap_rankings = zeros(n_features, n_bootstrap);

% Progress bar 표시
fprintf('Bootstrap 진행 중: ');

for b = 1:n_bootstrap
    % 복원추출로 재샘플링
    bootstrap_idx = randsample(n_samples, n_samples, true);
    X_boot = X_final(bootstrap_idx, :);
    y_boot = y_final(bootstrap_idx);

    % 정규화
    X_boot_norm = zscore(X_boot);

    % 샘플 가중치 재계산
    n_high_boot = sum(y_boot == 1);
    n_low_boot = sum(y_boot == 0);

    % 클래스가 하나만 있는 경우 스킵
    if n_high_boot == 0 || n_low_boot == 0
        bootstrap_weights(:, b) = NaN;
        bootstrap_rankings(:, b) = NaN;
        continue;
    end

    class_weights_boot = [n_samples/(2*n_low_boot), n_samples/(2*n_high_boot)];
    sample_weights_boot = zeros(size(y_boot));
    sample_weights_boot(y_boot == 0) = class_weights_boot(1);
    sample_weights_boot(y_boot == 1) = class_weights_boot(2);

    % Cost-Sensitive 모델 학습 (최적 Lambda 사용)
    try
        mdl_boot = fitclinear(X_boot_norm, y_boot, ...
            'Learner', 'logistic', ...
            'Cost', cost_matrix, ...
            'Weights', sample_weights_boot, ...
            'Regularization', 'ridge', ...
            'Lambda', optimal_lambda);

        % 가중치 추출 및 저장
        coefs = mdl_boot.Beta;
        positive_coefs = max(0, coefs);
        if sum(positive_coefs) > 0
            weights = positive_coefs / sum(positive_coefs) * 100;
        else
            weights = zeros(size(positive_coefs));
        end
        bootstrap_weights(:, b) = weights;

        % 순위 저장
        [~, ranks] = sort(weights, 'descend');
        for r = 1:length(ranks)
            bootstrap_rankings(ranks(r), b) = r;
        end

    catch
        % 실패한 경우 NaN 처리
        bootstrap_weights(:, b) = NaN;
        bootstrap_rankings(:, b) = NaN;
    end

    % Progress 표시
    if mod(b, 100) == 0
        fprintf('.');
    end
end
fprintf(' 완료!\n\n');

% Bootstrap 통계 계산
fprintf('【Bootstrap 결과 분석】\n');
fprintf('────────────────────────────────────────────\n\n');

% 각 역량별 통계
bootstrap_stats = table();
bootstrap_stats.Feature = feature_names';
bootstrap_stats.Original_Weight = final_weights;
bootstrap_stats.Boot_Mean = nanmean(bootstrap_weights, 2);
bootstrap_stats.Boot_Std = nanstd(bootstrap_weights, 0, 2);
bootstrap_stats.CI_Lower = prctile(bootstrap_weights, 2.5, 2);
bootstrap_stats.CI_Upper = prctile(bootstrap_weights, 97.5, 2);
bootstrap_stats.CV = bootstrap_stats.Boot_Std ./ (bootstrap_stats.Boot_Mean + eps);  % 변동계수

% 순위 안정성
bootstrap_stats.Avg_Rank = zeros(n_features, 1);
bootstrap_stats.Top3_Prob = zeros(n_features, 1);
bootstrap_stats.Top5_Prob = zeros(n_features, 1);

for i = 1:n_features
    valid_ranks = bootstrap_rankings(i, :);
    valid_ranks = valid_ranks(~isnan(valid_ranks));

    if ~isempty(valid_ranks)
        bootstrap_stats.Avg_Rank(i) = mean(valid_ranks);
        bootstrap_stats.Top3_Prob(i) = sum(valid_ranks <= 3) / length(valid_ranks) * 100;
        bootstrap_stats.Top5_Prob(i) = sum(valid_ranks <= 5) / length(valid_ranks) * 100;
    end
end

% 원본 가중치 기준으로 정렬
bootstrap_stats = sortrows(bootstrap_stats, 'Original_Weight', 'descend');

% 결과 출력
fprintf('가중치 안정성 분석 (상위 10개):\n');
fprintf('%-20s | 원본(%%) | 평균(%%) | 95%% CI | CV | Top3확률(%%) | Top5확률(%%)\n', '역량');
fprintf('%s\n', repmat('-', 95, 1));

for i = 1:min(10, height(bootstrap_stats))
    fprintf('%-20s | %7.2f | %7.2f | [%5.2f-%5.2f] | %4.2f | %7.1f | %7.1f\n', ...
        bootstrap_stats.Feature{i}, ...
        bootstrap_stats.Original_Weight(i), ...
        bootstrap_stats.Boot_Mean(i), ...
        bootstrap_stats.CI_Lower(i), ...
        bootstrap_stats.CI_Upper(i), ...
        bootstrap_stats.CV(i), ...
        bootstrap_stats.Top3_Prob(i), ...
        bootstrap_stats.Top5_Prob(i));
end

% 안정성 평가
fprintf('\n【안정성 평가】\n');
fprintf('────────────────────────────────────────────\n');

% 매우 안정적 (CV < 0.3 & Top3 > 70%)
very_stable = bootstrap_stats.CV < 0.3 & bootstrap_stats.Top3_Prob > 70;
if any(very_stable)
    fprintf('✅ 매우 안정적인 역량 (일관되게 중요):\n');
    stable_features = find(very_stable);
    for i = stable_features'
        fprintf('   - %s (CV=%.2f, Top3=%.1f%%)\n', ...
            bootstrap_stats.Feature{i}, ...
            bootstrap_stats.CV(i), ...
            bootstrap_stats.Top3_Prob(i));
    end
else
    fprintf('✅ 매우 안정적인 역량: 없음\n');
end

% 불안정 (CV > 0.5 | Top5 < 30%)
unstable = bootstrap_stats.CV > 0.5 | bootstrap_stats.Top5_Prob < 30;
if any(unstable)
    fprintf('\n⚠️ 불안정한 역량 (해석 주의):\n');
    unstable_features = find(unstable);
    for i = unstable_features'
        fprintf('   - %s (CV=%.2f, Top5=%.1f%%)\n', ...
            bootstrap_stats.Feature{i}, ...
            bootstrap_stats.CV(i), ...
            bootstrap_stats.Top5_Prob(i));
    end
else
    fprintf('\n⚠️ 불안정한 역량: 없음\n');
end

% Bootstrap 시각화
bootstrap_fig = figure('Position', [100, 100, 1600, 1200], 'Color', 'white');

% 전체 10개 역량의 Bootstrap 분포
subplot(3, 2, [1:4]);
top_10 = bootstrap_stats(1:min(10, height(bootstrap_stats)), :);
top_10_indices = zeros(height(top_10), 1);
for i = 1:height(top_10)
    top_10_indices(i) = find(strcmp(feature_names, top_10.Feature{i}));
end

boxplot(bootstrap_weights(top_10_indices, :)', ...
    'Labels', top_10.Feature, 'Colors', lines(10));
hold on;
% 원본 가중치 표시
for i = 1:height(top_10)
    feat_idx = top_10_indices(i);
    plot(i, final_weights(feat_idx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
end
ylabel('가중치 (%)', 'FontWeight', 'bold');
title('Bootstrap 가중치 분포 (전체 10개 역량)', 'FontWeight', 'bold');
legend({'원본 가중치'}, 'Location', 'northeast');
grid on;
% X축 레이블 회전 (가독성 향상)
xtickangle(45);

% 순위 변동성 히트맵
subplot(3, 2, [5:6]);
% 상위 10개 역량의 순위 확률 매트릭스
rank_prob_matrix = zeros(min(10, n_features), 10);
for i = 1:min(10, n_features)
    feat_idx = find(strcmp(feature_names, bootstrap_stats.Feature{i}));
    if ~isempty(feat_idx)
        for r = 1:10
            valid_ranks = bootstrap_rankings(feat_idx, :);
            valid_ranks = valid_ranks(~isnan(valid_ranks));
            if ~isempty(valid_ranks)
                rank_prob_matrix(i, r) = sum(valid_ranks == r) / length(valid_ranks) * 100;
            end
        end
    end
end

imagesc(rank_prob_matrix);
colormap(hot);
colorbar;
set(gca, 'YTick', 1:min(10, height(bootstrap_stats)), ...
         'YTickLabel', bootstrap_stats.Feature(1:min(10, height(bootstrap_stats))));
set(gca, 'XTick', 1:10, 'XTickLabel', 1:10);
xlabel('순위', 'FontWeight', 'bold');
ylabel('역량', 'FontWeight', 'bold');
title('순위 확률 분포 (빨간색 = 높은 확률)', 'FontWeight', 'bold');

sgtitle('Bootstrap 안정성 검증 (1000회 재샘플링)', 'FontSize', 16, 'FontWeight', 'bold');

% 그래프 저장
bootstrap_chart_filename = sprintf('bootstrap_stability_%s.png', datestr(now, 'yyyy-mm-dd_HHMMSS'));
saveas(bootstrap_fig, bootstrap_chart_filename);

fprintf('\n✅ Bootstrap 검증 완료\n');
fprintf('📊 시각화 저장 완료: %s\n', bootstrap_chart_filename);

%% 4.9 극단 그룹 비교 분석
fprintf('\n【STEP 17】 극단 그룹 t-test 비교 분석\n');
fprintf('────────────────────────────────────────────\n\n');

% 극단 그룹 정의 (가장 확실한 케이스만)
extreme_high = {'자연성', '성실한 가연성'};  % CODE 8, 7
extreme_low = {'무능한 불연성', '소화성'};   % CODE 2, 1

% 극단 그룹 인덱스 - 최종 분석 데이터에서 찾기
final_talent_types = matched_talent_types(final_idx);
final_talent_types = final_talent_types(complete_cases);

extreme_high_idx = ismember(final_talent_types, extreme_high);
extreme_low_idx = ismember(final_talent_types, extreme_low);

% 극단 그룹 데이터
X_extreme_high = X_final(extreme_high_idx, :);
X_extreme_low = X_final(extreme_low_idx, :);

fprintf('극단 그룹 구성:\n');
fprintf('  최고 성과자 (자연성, 성실한 가연성): %d명\n', sum(extreme_high_idx));
fprintf('  최저 성과자 (무능한 불연성, 소화성): %d명\n\n', sum(extreme_low_idx));

% 극단 그룹이 충분한지 확인
if sum(extreme_high_idx) < 3 || sum(extreme_low_idx) < 3
    fprintf('⚠️ 극단 그룹 샘플 수가 부족합니다. 분석을 건너뜁니다.\n');
else
    % t-test 결과 테이블 생성
    ttest_results = table();
    ttest_results.Feature = feature_names';
    ttest_results.High_Mean = zeros(n_features, 1);
    ttest_results.High_Std = zeros(n_features, 1);
    ttest_results.Low_Mean = zeros(n_features, 1);
    ttest_results.Low_Std = zeros(n_features, 1);
    ttest_results.Mean_Diff = zeros(n_features, 1);
    ttest_results.t_statistic = zeros(n_features, 1);
    ttest_results.p_value = zeros(n_features, 1);
    ttest_results.Cohen_d = zeros(n_features, 1);
    ttest_results.Significance = cell(n_features, 1);

    % 각 역량별 t-test 수행
    for i = 1:n_features
        high_scores = X_extreme_high(:, i);
        low_scores = X_extreme_low(:, i);

        % 기술통계
        ttest_results.High_Mean(i) = mean(high_scores);
        ttest_results.High_Std(i) = std(high_scores);
        ttest_results.Low_Mean(i) = mean(low_scores);
        ttest_results.Low_Std(i) = std(low_scores);
        ttest_results.Mean_Diff(i) = ttest_results.High_Mean(i) - ttest_results.Low_Mean(i);

        % t-test
        try
            [h, p, ci, stats] = ttest2(high_scores, low_scores);
            ttest_results.t_statistic(i) = stats.tstat;
            ttest_results.p_value(i) = p;
        catch
            ttest_results.t_statistic(i) = NaN;
            ttest_results.p_value(i) = NaN;
        end

        % Cohen's d (효과 크기)
        pooled_std = sqrt(((length(high_scores)-1)*var(high_scores) + ...
                          (length(low_scores)-1)*var(low_scores)) / ...
                          (length(high_scores) + length(low_scores) - 2));
        if pooled_std > 0
            ttest_results.Cohen_d(i) = ttest_results.Mean_Diff(i) / pooled_std;
        else
            ttest_results.Cohen_d(i) = 0;
        end

        % 유의성 표시
        p = ttest_results.p_value(i);
        if isnan(p)
            ttest_results.Significance{i} = 'NA';
        elseif p < 0.001
            ttest_results.Significance{i} = '***';
        elseif p < 0.01
            ttest_results.Significance{i} = '**';
        elseif p < 0.05
            ttest_results.Significance{i} = '*';
        elseif p < 0.1
            ttest_results.Significance{i} = '†';
        else
            ttest_results.Significance{i} = '';
        end
    end

    % Cohen's d 기준으로 정렬
    ttest_results = sortrows(ttest_results, 'Cohen_d', 'descend');

    % 결과 출력
    fprintf('【극단 그룹 비교 결과】\n');
    fprintf('────────────────────────────────────────────\n');
    fprintf('%-20s | 고성과(M±SD) | 저성과(M±SD) | 차이 | t값 | p값 | Cohen''s d | 효과\n', '역량');
    fprintf('%s\n', repmat('-', 105, 1));

    for i = 1:height(ttest_results)
        % 효과 크기 해석
        d = abs(ttest_results.Cohen_d(i));
        if d < 0.2
            effect = '무시';
        elseif d < 0.5
            effect = '작음';
        elseif d < 0.8
            effect = '중간';
        else
            effect = '큼';
        end

        fprintf('%-20s | %5.1f±%4.1f | %5.1f±%4.1f | %+5.1f | %+5.2f | %.3f%s | %+6.3f | %s\n', ...
            ttest_results.Feature{i}, ...
            ttest_results.High_Mean(i), ttest_results.High_Std(i), ...
            ttest_results.Low_Mean(i), ttest_results.Low_Std(i), ...
            ttest_results.Mean_Diff(i), ...
            ttest_results.t_statistic(i), ...
            ttest_results.p_value(i), ttest_results.Significance{i}, ...
            ttest_results.Cohen_d(i), effect);
    end

    % 유의한 차이를 보이는 역량만 추출 (p < 0.05 & |d| > 0.5)
    valid_p = ~isnan(ttest_results.p_value);
    significant_features = valid_p & ttest_results.p_value < 0.05 & abs(ttest_results.Cohen_d) > 0.5;

    fprintf('\n【핵심 차별화 역량】 (p<0.05 & Cohen''s d>0.5)\n');
    fprintf('────────────────────────────────────────────\n');
    if any(significant_features)
        sig_table = ttest_results(significant_features, :);
        for i = 1:height(sig_table)
            fprintf('• %s: 평균차이 %.1f점, Cohen''s d = %.2f\n', ...
                sig_table.Feature{i}, sig_table.Mean_Diff(i), sig_table.Cohen_d(i));
        end
    else
        fprintf('통계적으로 유의하고 실질적 효과가 큰 역량이 없습니다.\n');
    end

    % Bonferroni 보정
    bonferroni_alpha = 0.05 / n_features;
    bonferroni_sig = valid_p & ttest_results.p_value < bonferroni_alpha;

    fprintf('\n【Bonferroni 보정 후】 (α = %.4f)\n', bonferroni_alpha);
    fprintf('────────────────────────────────────────────\n');
    if any(bonferroni_sig)
        bon_table = ttest_results(bonferroni_sig, :);
        for i = 1:height(bon_table)
            fprintf('• %s: 여전히 유의함 (p = %.4f)\n', ...
                bon_table.Feature{i}, bon_table.p_value(i));
        end
    else
        fprintf('다중비교 보정 후 유의한 역량이 없습니다.\n');
    end

    % 시각화
    extreme_fig = figure('Position', [100, 100, 1200, 600], 'Color', 'white');

    % Cohen's d 막대그래프
    subplot(1, 2, 1);
    bar(ttest_results.Cohen_d, 'FaceColor', [0.2, 0.4, 0.8]);
    hold on;
    % 효과 크기 기준선
    yline(0.8, '--r', 'LineWidth', 1.5);
    yline(0.5, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5);
    yline(-0.5, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5);
    yline(-0.8, '--r', 'LineWidth', 1.5);
    set(gca, 'XTick', 1:height(ttest_results), ...
             'XTickLabel', ttest_results.Feature, ...
             'XTickLabelRotation', 45);
    ylabel('Cohen''s d', 'FontWeight', 'bold');
    title('효과 크기 (극단 그룹 차이)', 'FontWeight', 'bold');
    legend({'Cohen''s d', '큰 효과(0.8)', '중간 효과(0.5)'}, 'Location', 'best');
    grid on;

    % p-value 비교
    subplot(1, 2, 2);
    valid_p_values = ttest_results.p_value;
    valid_p_values(isnan(valid_p_values)) = 1;  % NaN을 1로 대체
    bar(-log10(valid_p_values), 'FaceColor', [0.8, 0.3, 0.3]);
    hold on;
    % 유의수준 선
    yline(-log10(0.05), '--g', 'LineWidth', 2);
    yline(-log10(0.01), '--', 'Color', [1, 0.5, 0], 'LineWidth', 2);
    yline(-log10(0.001), '--r', 'LineWidth', 2);
    set(gca, 'XTick', 1:height(ttest_results), ...
             'XTickLabel', ttest_results.Feature, ...
             'XTickLabelRotation', 45);
    ylabel('-log10(p-value)', 'FontWeight', 'bold');
    title('통계적 유의성', 'FontWeight', 'bold');
    legend({'p-value', 'p<0.05', 'p<0.01', 'p<0.001'}, 'Location', 'best');
    grid on;

    sgtitle('극단 그룹 t-test 분석 결과', 'FontSize', 16, 'FontWeight', 'bold');

    % 저장
    extreme_chart_filename = sprintf('extreme_group_ttest_%s.png', datestr(now, 'yyyy-mm-dd_HHMMSS'));
    saveas(extreme_fig, extreme_chart_filename);

    fprintf('\n✅ 극단 그룹 분석 완료\n');
    fprintf('📊 시각화 저장 완료: %s\n', extreme_chart_filename);

    % 결과를 파일에 저장
    result_data.ttest_results = ttest_results;
    result_data.extreme_analysis = struct(...
        'extreme_high', {extreme_high}, ...
        'extreme_low', {extreme_low}, ...
        'n_high', sum(extreme_high_idx), ...
        'n_low', sum(extreme_low_idx));
end

% Bootstrap 결과도 파일에 저장
result_data.bootstrap_stats = bootstrap_stats;
result_data.bootstrap_weights = bootstrap_weights;
result_data.bootstrap_rankings = bootstrap_rankings;
save(weight_filename, 'result_data', 'weight_results_significant', 'bootstrap_stats');

%% ========================================================================
%                          최종 가중치 정리 및 산출 방법
% =========================================================================
fprintf('\n\n╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║                    최종 가중치 정리 및 활용법             ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

fprintf('【STEP 18】 최종 가중치 정리 및 산출 방법\n');
fprintf('────────────────────────────────────────────\n');

%% 18.1 최종 가중치 요약 테이블 생성
fprintf('\n▶ 최종 가중치 요약 테이블 생성\n');

% 가중치 요약 테이블 생성 (상위 15개만)
final_summary = table();
top_15_idx = min(15, height(weight_results_significant));
final_summary.Rank = (1:top_15_idx)';
final_summary.Competency = weight_results_significant.Feature(1:top_15_idx);
final_summary.Weight_Percent = round(weight_results_significant.Weight_Percent(1:top_15_idx), 2);
final_summary.Raw_Coefficient = round(weight_results_significant.Raw_Coefficient(1:top_15_idx), 4);

% Bootstrap 안정성 정보 추가 (있는 경우)
if exist('bootstrap_stats', 'var') && ~isempty(bootstrap_stats)
    final_summary.Bootstrap_CV = zeros(top_15_idx, 1);
    final_summary.Bootstrap_Stability = cell(top_15_idx, 1);
    
    for i = 1:top_15_idx
        comp_name = final_summary.Competency{i};
        boot_idx = strcmp(bootstrap_stats.Feature, comp_name);
        if any(boot_idx)
            cv_val = bootstrap_stats.CV(boot_idx);
            final_summary.Bootstrap_CV(i) = round(cv_val, 3);
            
            if cv_val < 0.3
                final_summary.Bootstrap_Stability{i} = 'Very Stable';
            elseif cv_val < 0.5
                final_summary.Bootstrap_Stability{i} = 'Stable';
            elseif cv_val < 0.7
                final_summary.Bootstrap_Stability{i} = 'Moderate';
            else
                final_summary.Bootstrap_Stability{i} = 'Unstable';
            end
        else
            final_summary.Bootstrap_CV(i) = NaN;
            final_summary.Bootstrap_Stability{i} = 'N/A';
        end
    end
end

% 극단 그룹 t-test 정보 추가 (있는 경우)
if exist('ttest_results', 'var')
    final_summary.Cohens_d = zeros(top_15_idx, 1);
    final_summary.P_Value = zeros(top_15_idx, 1);
    final_summary.Statistical_Significance = cell(top_15_idx, 1);
    
    for i = 1:top_15_idx
        comp_name = final_summary.Competency{i};
        ttest_idx = strcmp(ttest_results.Feature, comp_name);
        if any(ttest_idx)
            final_summary.Cohens_d(i) = round(ttest_results.Cohen_d(ttest_idx), 3);
            final_summary.P_Value(i) = round(ttest_results.p_value(ttest_idx), 4);
            
            p_val = ttest_results.p_value(ttest_idx);
            if p_val < 0.001
                final_summary.Statistical_Significance{i} = 'p<0.001***';
            elseif p_val < 0.01
                final_summary.Statistical_Significance{i} = 'p<0.01**';
            elseif p_val < 0.05
                final_summary.Statistical_Significance{i} = 'p<0.05*';
            elseif p_val < 0.1
                final_summary.Statistical_Significance{i} = 'p<0.1†';
            else
                final_summary.Statistical_Significance{i} = 'n.s.';
            end
        else
            final_summary.Cohens_d(i) = NaN;
            final_summary.P_Value(i) = NaN;
            final_summary.Statistical_Significance{i} = 'N/A';
        end
    end
end

% 종합 신뢰도 등급 계산
final_summary.Reliability_Grade = cell(top_15_idx, 1);
for i = 1:top_15_idx
    score = 0;
    
    % 가중치 점수 (0-3점)
    if final_summary.Weight_Percent(i) >= 20
        score = score + 3;
    elseif final_summary.Weight_Percent(i) >= 15
        score = score + 2.5;
    elseif final_summary.Weight_Percent(i) >= 10
        score = score + 2;
    elseif final_summary.Weight_Percent(i) >= 5
        score = score + 1;
    end
    
    % Bootstrap 안정성 점수 (0-2점)
    if exist('bootstrap_stats', 'var') && ~isnan(final_summary.Bootstrap_CV(i))
        if final_summary.Bootstrap_CV(i) < 0.3
            score = score + 2;
        elseif final_summary.Bootstrap_CV(i) < 0.5
            score = score + 1.5;
        elseif final_summary.Bootstrap_CV(i) < 0.7
            score = score + 1;
        end
    end
    
    % 통계적 유의성 점수 (0-2점)
    if exist('ttest_results', 'var') && ~isnan(final_summary.P_Value(i))
        if final_summary.P_Value(i) < 0.001
            score = score + 2;
        elseif final_summary.P_Value(i) < 0.01
            score = score + 1.5;
        elseif final_summary.P_Value(i) < 0.05
            score = score + 1;
        end
    end
    
    % 등급 부여 (총 7점 만점)
    if score >= 6
        final_summary.Reliability_Grade{i} = 'A+ (매우 신뢰)';
    elseif score >= 5
        final_summary.Reliability_Grade{i} = 'A (신뢰)';
    elseif score >= 4
        final_summary.Reliability_Grade{i} = 'B (보통)';
    elseif score >= 3
        final_summary.Reliability_Grade{i} = 'C (주의)';
    else
        final_summary.Reliability_Grade{i} = 'D (불안정)';
    end
end

%% 18.2 결과 출력
fprintf('\n【최종 역량 가중치 요약표】\n');
fprintf('═══════════════════════════════════════════════════════════════════════════════\n');
fprintf('순위 | %-25s | 가중치(%%) | 계수     | 안정성   | 유의성    | 신뢰등급\n', '역량명');
fprintf('─────┼───────────────────────────┼──────────┼─────────┼─────────┼─────────┼─────────\n');

for i = 1:top_15_idx
    fprintf('%2d   | %-25s | %8.2f | %8.4f | %-8s | %-9s | %s\n', ...
        final_summary.Rank(i), ...
        final_summary.Competency{i}, ...
        final_summary.Weight_Percent(i), ...
        final_summary.Raw_Coefficient(i), ...
        final_summary.Bootstrap_Stability{i}, ...
        final_summary.Statistical_Significance{i}, ...
        final_summary.Reliability_Grade{i});
end
fprintf('═══════════════════════════════════════════════════════════════════════════════\n');

%% 18.3 가중치 산출 공식 및 활용법
fprintf('\n【가중치 산출 공식 및 활용법】\n');
fprintf('────────────────────────────────────────────\n');

fprintf('\n■ 1. 개인별 종합점수 계산 공식:\n');
fprintf('   종합점수 = Σ(표준화된_역량점수i × 가중치i) / 100\n\n');

fprintf('■ 2. 표준화 공식 (Z-score):\n');
fprintf('   표준화점수 = (원점수 - 평균) / 표준편차\n\n');

fprintf('■ 3. 고성과자 판별 기준:\n');
fprintf('   판별임계값: %.4f\n', optimal_threshold);
fprintf('   → 종합점수 > %.4f 이면 고성과자 가능성 높음\n\n', optimal_threshold);

fprintf('■ 4. 신뢰도별 활용 가이드:\n');
fprintf('   A+ 등급: 핵심 선발/승진 기준으로 활용\n');
fprintf('   A  등급: 중요한 평가 요소로 활용\n');
fprintf('   B  등급: 참고 지표로 활용\n');
fprintf('   C  등급: 보조 지표로만 활용\n');
fprintf('   D  등급: 활용 주의 (추가 검증 필요)\n\n');

%% 18.4 실무 활용 예시 코드 생성
fprintf('■ 5. 실무 활용 MATLAB 코드 예시:\n');

% 상위 10개 개수 계산
top_10_count = min(10, height(weight_results_significant));

% 실무 활용 함수 생성
function_code = sprintf([
'function [score, prediction, probability] = calculateTalentScore(competency_scores)\n'...
'%% 역량 점수를 입력받아 종합점수와 예측 결과를 반환하는 함수\n'...
'%% Input: competency_scores - 역량 점수 벡터 (1x%d)\n'...
'%% Output: score - 종합점수, prediction - 예측결과, probability - 확률\n\n'...
'    %% 가중치 정보 (상위 10개 핵심 역량)\n'...
'    weights = ['], top_10_count);

% 상위 10개 가중치만 포함
top_10_weights = weight_results_significant.Weight_Percent(1:top_10_count);
for i = 1:length(top_10_weights)
    if i == length(top_10_weights)
        function_code = [function_code, sprintf('%.4f', top_10_weights(i))];
    else
        function_code = [function_code, sprintf('%.4f, ', top_10_weights(i))];
    end
end

function_code = [function_code, sprintf([
'];\n'...
'    feature_names = {'])];

% 상위 10개 역량명만 포함
if istable(weight_results_significant)
    top_10_names = weight_results_significant.Feature(1:top_10_count);
else
    top_10_names = feature_names(1:top_10_count);
end
for i = 1:length(top_10_names)
    if i == length(top_10_names)
        function_code = [function_code, sprintf('''%s''', top_10_names{i})];
    else
        function_code = [function_code, sprintf('''%s'', ', top_10_names{i})];
    end
end

function_code = [function_code, sprintf([
'};\n'...
'    threshold = %.4f;\n\n'...
'    %% 입력 검증\n'...
'    if length(competency_scores) ~= length(weights)\n'...
'        error(''입력 역량 점수 개수가 맞지 않습니다. %%d개 필요'', length(weights));\n'...
'    end\n\n'...
'    %% 표준화 (Z-score)\n'...
'    normalized_scores = (competency_scores - mean(competency_scores)) / std(competency_scores);\n\n'...
'    %% 가중 점수 계산\n'...
'    score = sum(normalized_scores .* (weights/100));\n\n'...
'    %% 예측 및 확률 계산\n'...
'    if score > threshold\n'...
'        prediction = ''고성과자 가능성 높음'';\n'...
'    else\n'...
'        prediction = ''일반 성과자'';\n'...
'    end\n'...
'    \n'...
'    %% 시그모이드 함수로 확률 계산\n'...
'    probability = 1 / (1 + exp(-(score - threshold)));\n'...
'end\n\n'...
'%% 사용 예시:\n'...
'%% new_scores = [75, 82, 68, 90, 77, 85, 70, 88, 79, 81];  %% 10개 역량 점수\n'...
'%% [score, pred, prob] = calculateTalentScore(new_scores);\n'...
'%% fprintf(''종합점수: %%.2f, 예측: %%s, 확률: %%.1f%%%%\\n'', score, pred, prob*100);\n'
], optimal_threshold)];

fprintf('\n   → 실무용 함수가 생성되었습니다.\n');

% 함수를 별도 파일로 저장
func_filename = fullfile(config.output_dir, 'calculateTalentScore.m');
try
    fid = fopen(func_filename, 'w');
    fprintf(fid, '%s', function_code);
    fclose(fid);
    fprintf('   → 함수 파일 저장: %s\n', func_filename);
catch
    fprintf('   ⚠ 함수 파일 저장 실패\n');
end

%% 18.5 검증 예시 실행
fprintf('\n■ 6. 검증 예시 (기존 데이터로 테스트):\n');

if exist('X_final', 'var') && exist('y_final', 'var')
    % 랜덤하게 3명 선택하여 테스트
    test_indices = randsample(length(y_final), min(3, length(y_final)));
    
    fprintf('   실제 데이터 검증 결과:\n');
    fprintf('   ─────────────────────────────────────\n');
    
    for i = 1:length(test_indices)
        idx = test_indices(i);
        actual_scores = X_final(idx, 1:min(10, size(X_final, 2)));
        actual_label = y_final(idx);
        
        % 표준화
        norm_scores = (actual_scores - mean(actual_scores)) / std(actual_scores);
        
        % 가중 점수 계산
        weighted_score = sum(norm_scores .* (top_10_weights'/100));
        weighted_score = weighted_score(1);  % 스칼라로 변환
        
        % 예측
        if weighted_score > optimal_threshold
            prediction = '고성과자';
        else
            prediction = '일반성과자';
        end
        
        % 확률
        probability = 1 / (1 + exp(-(weighted_score - optimal_threshold)));
        
        % 실제 라벨
        if actual_label == 1
            actual = '고성과자';
        else
            actual = '저성과자';
        end
        
        fprintf('   샘플 %d: 점수=%.3f, 예측=%s, 확률=%.1f%%, 실제=%s\n', ...
            i, weighted_score, prediction, probability*100, actual);
    end
end

fprintf('\n✅ 가중치 정리 및 산출 방법 완료!\n');
fprintf('📁 실무용 함수: calculateTalentScore.m 생성됨\n\n');

%% ========================================================================
%                    엑셀 결과 보고서 생성 및 파일 관리
% =========================================================================
fprintf('\n〔STEP 15〕 엑셀 결과 보고서 생성\n');
fprintf('────────────────────────────────────────────\n');

% 기존 엑셀 파일 정리 (최신 파일만 유지)
excel_pattern = fullfile(config.output_dir, '*TalentType_Analysis_Results*.xlsx');
existing_excel_files = dir(excel_pattern);

if ~isempty(existing_excel_files)
    fprintf('기존 엑셀 파일 정리 중...\n');

    % 파일들을 날짜순으로 정렬
    [~, sort_idx] = sort([existing_excel_files.datenum], 'descend');
    existing_excel_files = existing_excel_files(sort_idx);

    % 가장 최신 파일 제외하고 나머지 삭제
    for i = 2:length(existing_excel_files)
        old_file = fullfile(existing_excel_files(i).folder, existing_excel_files(i).name);
        try
            delete(old_file);
            fprintf('  ✓ 기존 파일 삭제: %s\n', existing_excel_files(i).name);
        catch
            fprintf('  ⚠ 파일 삭제 실패: %s\n', existing_excel_files(i).name);
        end
    end
end

% 엑셀 파일명 생성 (비활성/과활성 포함 모델) - 영문 파일명 사용
excel_filename = fullfile(config.output_dir, sprintf('TalentType_Analysis_Results_Filtered_%s.xlsx', config.timestamp));

try
    fprintf('엑셀 보고서 생성 중...\n');

    %% 1. 요약 정보 시트
    summary_data = {
        'Item', 'Value', 'Description';
        'Analysis_Date', datestr(now, 'yyyy-mm-dd HH:MM:SS'), 'Report generation time';
        'Model_Type', 'Cost-Sensitive Learning (Including Inactive/Overactive)', 'Model method used';
        'Total_Participants', height(matched_comp), 'Number of competency test participants';
        'High_Performers', sum(y_weight == 1), 'Number of high performers in analysis';
        'Low_Performers', sum(y_weight == 0), 'Number of low performers in analysis';
        'Original_Competencies', width(comp_upper), 'Number of original competency items';
        'Filtered_Competencies', width(matched_comp), 'Number of items including inactive/overactive';
        'Special_Items_Included', 'Inactive, Overactive', 'Inactive/Overactive items included';
        'Optimal_Lambda', optimal_lambda, 'L2 regularization coefficient selected by LOO-CV';
        'CV_AUC', best_auc, 'Leave-One-Out cross-validation AUC score';
        'CV_Accuracy', cv_scores(best_idx), 'Leave-One-Out cross-validation accuracy';
        'Cohens_d', cohens_d, 'Effect size (High vs Low performers)';
        'Misclassification_Cost_Ratio', '1.5:1', 'Cost of misclassifying low performers as high performers'
    };

    writecell(summary_data, excel_filename, 'Sheet', '1_Analysis_Summary');
    fprintf('  ✓ 분석 요약 시트 생성\n');

    %% 2. 역량 가중치 결과 (주요 결과)
    weights_table = weight_results_significant;
    weights_table.Properties.VariableNames = {'Competency_Name', 'Weight_Percent', 'Raw_Coefficient'};

    % 순위 추가
    weights_table = addvars(weights_table, (1:height(weights_table))', 'Before', 'Competency_Name', 'NewVariableNames', 'Rank');

    % 해석 추가
    interpretation = cell(height(weights_table), 1);
    for i = 1:height(weights_table)
        weight_val = weights_table.Weight_Percent(i);
        if weight_val >= 20
            interpretation{i} = 'Very Important (≥20%)';
        elseif weight_val >= 15
            interpretation{i} = 'Important (15-20%)';
        elseif weight_val >= 10
            interpretation{i} = 'Moderate (10-15%)';
        elseif weight_val >= 5
            interpretation{i} = 'Basic (5-10%)';
        else
            interpretation{i} = 'Low (<5%)';
        end
    end
    weights_table = addvars(weights_table, interpretation, 'NewVariableNames', 'Importance_Level');

    writetable(weights_table, excel_filename, 'Sheet', '2_Competency_Weights', 'WriteVariableNames', true);
    fprintf('  ✓ 역량 가중치 시트 생성\n');
    
    %% 2-1. 최종 요약 테이블 추가 (신규)
    if exist('final_summary', 'var')
        writetable(final_summary, excel_filename, 'Sheet', '2_1_Final_Summary', 'WriteVariableNames', true);
        fprintf('  ✓ 최종 요약 테이블 시트 생성\n');
    end

    %% 3. Bootstrap 검증 결과
    if exist('bootstrap_stats', 'var') && ~isempty(bootstrap_stats)
        bootstrap_table = table();
        bootstrap_table.Competency_Name = bootstrap_stats.Feature;
        bootstrap_table.Mean_Weight = bootstrap_stats.Boot_Mean;
        bootstrap_table.Std_Weight = bootstrap_stats.Boot_Std;
        bootstrap_table.CI_Lower = bootstrap_stats.CI_Lower;
        bootstrap_table.CI_Upper = bootstrap_stats.CI_Upper;
        bootstrap_table.CV = bootstrap_stats.CV;

        % 안정성 해석
        stability_interp = cell(height(bootstrap_table), 1);
        for i = 1:height(bootstrap_table)
            cv_val = bootstrap_table.CV(i);
            if cv_val < 0.3
                stability_interp{i} = 'Very Stable';
            elseif cv_val < 0.5
                stability_interp{i} = 'Stable';
            elseif cv_val < 0.7
                stability_interp{i} = 'Moderate';
            else
                stability_interp{i} = 'Unstable';
            end
        end
        bootstrap_table = addvars(bootstrap_table, stability_interp, 'NewVariableNames', 'Stability_Assessment');

        writetable(bootstrap_table, excel_filename, 'Sheet', '3_Bootstrap_Validation', 'WriteVariableNames', true);
        fprintf('  ✓ Bootstrap 검증 시트 생성\n');
    end

    %% 4. 인재유형별 프로필 비교
    if exist('type_profiles', 'var') && exist('unique_matched_types', 'var')
        % 인재유형 프로필 테이블 생성
        profile_table = array2table(type_profiles, 'VariableNames', valid_comp_cols);
        profile_table = addvars(profile_table, unique_matched_types', 'Before', valid_comp_cols{1}, 'NewVariableNames', 'Talent_Type');

        % 전체 평균 추가
        overall_row = table();
        overall_row.Talent_Type = {'Overall_Average'};
        for j = 1:length(valid_comp_cols)
            overall_row.(valid_comp_cols{j}) = overall_mean_profile(j);
        end
        profile_table = [profile_table; overall_row];

        writetable(profile_table, excel_filename, 'Sheet', '4_Talent_Type_Profiles', 'WriteVariableNames', true);
        fprintf('  ✓ 인재유형별 프로필 시트 생성\n');
    end

    %% 5. 마할라노비스 거리 분석
    if exist('distance_matrix', 'var') && exist('unique_matched_types', 'var')
        % 거리 행렬을 테이블로 변환
        distance_table = array2table(distance_matrix, 'VariableNames', unique_matched_types, 'RowNames', unique_matched_types);
        distance_table = addvars(distance_table, unique_matched_types', 'Before', unique_matched_types{1}, 'NewVariableNames', 'Talent_Type');

        writetable(distance_table, excel_filename, 'Sheet', '5_Mahalanobis_Distance', 'WriteVariableNames', true);
        fprintf('  ✓ 마할라노비스 거리 시트 생성\n');
    end

    %% 6. 극단 그룹 비교 분석
    if exist('ttest_results', 'var')
        extreme_table = struct2table(ttest_results);

        % 통계적 유의성 해석 추가
        significance_interp = cell(height(extreme_table), 1);
        for i = 1:height(extreme_table)
            p_val = extreme_table.p_value(i);
            if p_val < 0.001
                significance_interp{i} = 'Highly Significant (p<0.001)';
            elseif p_val < 0.01
                significance_interp{i} = 'Significant (p<0.01)';
            elseif p_val < 0.05
                significance_interp{i} = 'Significant (p<0.05)';
            elseif p_val < 0.1
                significance_interp{i} = 'Borderline (p<0.1)';
            else
                significance_interp{i} = 'Not Significant (p≥0.1)';
            end
        end
        extreme_table = addvars(extreme_table, significance_interp, 'NewVariableNames', 'Significance_Assessment');

        writetable(extreme_table, excel_filename, 'Sheet', '6_Extreme_Group_Comparison', 'WriteVariableNames', true);
        fprintf('  ✓ 극단 그룹 비교 시트 생성\n');
    end

    %% 7. 모델 성능 지표
    if exist('cv_scores', 'var') && exist('cv_aucs', 'var')
        performance_data = {
            'Metric', 'Value', 'Description';
            'Optimal_Lambda', optimal_lambda, 'L2 regularization coefficient';
            'CV_Accuracy', cv_scores(best_idx), 'Leave-One-Out CV accuracy';
            'CV_AUC', best_auc, 'Leave-One-Out CV AUC';
            'Cohens_d', cohens_d, 'Effect size';
            'Class_High_Performers', sum(y_weight == 1), 'Number of high performers';
            'Class_Low_Performers', sum(y_weight == 0), 'Number of low performers';
            'Cost_Matrix_00', cost_matrix(1,1), 'Correct classification cost (Low→Low)';
            'Cost_Matrix_01', cost_matrix(1,2), 'Misclassification cost (Low→High)';
            'Cost_Matrix_10', cost_matrix(2,1), 'Misclassification cost (High→Low)';
            'Cost_Matrix_11', cost_matrix(2,2), 'Correct classification cost (High→High)'
        };

        writecell(performance_data, excel_filename, 'Sheet', '7_Model_Performance_Metrics');
        fprintf('  ✓ 모델 성능 지표 시트 생성\n');
    end

    fprintf('\n✓ 엑셀 보고서 생성 완료: %s\n', excel_filename);

catch excel_error
    warning('엑셀 파일 생성 실패: %s', excel_error.message);
    fprintf('⚠ 엑셀 내보내기에 실패했지만 분석은 완료되었습니다.\n');
end

fprintf('\n\n╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║                      분석 완료!                             ║\n');
fprintf('║                                                              ║\n');
fprintf('║  ✅ Cost-Sensitive Learning 기반 고성과자 예측 시스템 완료    ║\n');
fprintf('║  ✅ 클래스 불균형 해결 및 비용 행렬 적용 완료                 ║\n');
fprintf('║  ✅ Leave-One-Out 교차검증으로 최적 Lambda 선택 완료          ║\n');
fprintf('║  ✅ L2 정규화를 통한 모든 10개 역량 유지 완료                 ║\n');
fprintf('║  ✅ 저성과자→고성과자 오분류 비용 1.5배 적용 완료             ║\n');
fprintf('║  ✅ Bootstrap 검증으로 가중치 안정성 확인 완료                ║\n');
fprintf('║  ✅ 극단 그룹 t-test로 차별화 역량 확인 완료                  ║\n');
fprintf('║                                                              ║\n');
fprintf('║  📁 가중치 파일 저장됨: %s          ║\n', weight_filename);
fprintf('║  📊 Cost-Sensitive 차트: %s           ║\n', chart_filename);
fprintf('║  📊 Bootstrap 차트: %s      ║\n', bootstrap_chart_filename);
if exist('extreme_chart_filename', 'var')
    fprintf('║  📊 극단 그룹 차트: %s        ║\n', extreme_chart_filename);
end
if exist('excel_filename', 'var')
    [~, excel_name, excel_ext] = fileparts(excel_filename);
    fprintf('║  📋 Excel 결과 보고서: %s%s ║\n', excel_name, excel_ext);
end
fprintf('╚══════════════════════════════════════════════════════════════╝\n');




% 레이더 차트 생성 완료
