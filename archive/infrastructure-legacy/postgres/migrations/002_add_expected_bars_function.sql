-- Migration: Add function to calculate expected trading bars
-- Description: Calculate expected 5-minute bars for USD/COP based on trading hours
-- Trading Hours: Monday-Friday, 8:00 AM - 12:55 PM COT (60 bars per day)
-- Date: 2025-10-23

-- Function: Calculate expected trading days in a date range
CREATE OR REPLACE FUNCTION public.count_expected_trading_days(
    start_date DATE,
    end_date DATE
) RETURNS INTEGER AS $$
DECLARE
    trading_days INTEGER := 0;
    iter_date DATE;
    day_of_week INTEGER;
BEGIN
    -- Iterate through each day in the range
    iter_date := start_date;

    WHILE iter_date <= end_date LOOP
        -- Get day of week (1=Monday, 7=Sunday)
        day_of_week := EXTRACT(DOW FROM iter_date);

        -- Count if it's a weekday (Monday-Friday: 1-5)
        IF day_of_week BETWEEN 1 AND 5 THEN
            -- TODO: Add holiday exclusion logic here if needed
            -- For now, we count all weekdays
            trading_days := trading_days + 1;
        END IF;

        iter_date := iter_date + INTERVAL '1 day';
    END LOOP;

    RETURN trading_days;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: Calculate expected bars for a date range
CREATE OR REPLACE FUNCTION public.count_expected_bars(
    start_date DATE,
    end_date DATE,
    bars_per_day INTEGER DEFAULT 60
) RETURNS INTEGER AS $$
DECLARE
    trading_days INTEGER;
    expected_bars INTEGER;
BEGIN
    -- Get count of trading days in range
    trading_days := count_expected_trading_days(start_date, end_date);

    -- Calculate total expected bars
    expected_bars := trading_days * bars_per_day;

    RETURN expected_bars;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: Calculate completeness percentage for a symbol in date range
CREATE OR REPLACE FUNCTION public.calculate_completeness(
    p_symbol TEXT,
    p_start_date DATE,
    p_end_date DATE
) RETURNS TABLE(
    symbol TEXT,
    start_date DATE,
    end_date DATE,
    expected_bars INTEGER,
    actual_bars BIGINT,
    completeness_pct NUMERIC(5,2),
    trading_days INTEGER
) AS $$
DECLARE
    v_trading_days INTEGER;
    v_expected_bars INTEGER;
    v_actual_bars BIGINT;
    v_end_date DATE;
BEGIN
    -- Don't count future days - use min of p_end_date and CURRENT_DATE
    v_end_date := LEAST(p_end_date, CURRENT_DATE);

    -- Calculate expected trading days and bars (only up to today)
    v_trading_days := count_expected_trading_days(p_start_date, v_end_date);
    v_expected_bars := v_trading_days * 60; -- 60 bars per trading day

    -- Count actual bars in database for the symbol and date range
    -- Filter by trading hours: 8:00 AM - 12:55 PM COT
    -- Don't count future data
    SELECT COUNT(*) INTO v_actual_bars
    FROM usdcop_m5_ohlcv
    WHERE usdcop_m5_ohlcv.symbol = p_symbol
      AND DATE(time AT TIME ZONE 'America/Bogota') BETWEEN p_start_date AND v_end_date
      AND EXTRACT(HOUR FROM time AT TIME ZONE 'America/Bogota') BETWEEN 8 AND 12
      AND EXTRACT(DOW FROM time AT TIME ZONE 'America/Bogota') BETWEEN 1 AND 5;

    -- Return results (with original p_end_date for display)
    RETURN QUERY SELECT
        p_symbol,
        p_start_date,
        p_end_date,  -- Original end date requested
        v_expected_bars,
        v_actual_bars,
        CASE
            WHEN v_expected_bars > 0 THEN ROUND((v_actual_bars::NUMERIC / v_expected_bars * 100), 2)
            ELSE 0.0
        END,
        v_trading_days;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: Calculate weekly completeness for current week
CREATE OR REPLACE FUNCTION public.get_weekly_completeness(
    p_symbol TEXT DEFAULT 'USDCOP'
) RETURNS TABLE(
    symbol TEXT,
    week_start DATE,
    week_end DATE,
    expected_bars INTEGER,
    actual_bars BIGINT,
    completeness_pct NUMERIC(5,2),
    trading_days INTEGER
) AS $$
DECLARE
    v_week_start DATE;
    v_week_end DATE;
BEGIN
    -- Calculate start of current week (Monday)
    v_week_start := DATE_TRUNC('week', CURRENT_DATE AT TIME ZONE 'America/Bogota')::DATE;

    -- Calculate end of current week (Friday)
    v_week_end := (v_week_start + INTERVAL '4 days')::DATE;

    -- Return completeness for the week
    RETURN QUERY
    SELECT * FROM calculate_completeness(p_symbol, v_week_start, v_week_end);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: Calculate monthly completeness for current month
CREATE OR REPLACE FUNCTION public.get_monthly_completeness(
    p_symbol TEXT DEFAULT 'USDCOP'
) RETURNS TABLE(
    symbol TEXT,
    month_start DATE,
    month_end DATE,
    expected_bars INTEGER,
    actual_bars BIGINT,
    completeness_pct NUMERIC(5,2),
    trading_days INTEGER
) AS $$
DECLARE
    v_month_start DATE;
    v_month_end DATE;
BEGIN
    -- Calculate start and end of current month
    v_month_start := DATE_TRUNC('month', CURRENT_DATE AT TIME ZONE 'America/Bogota')::DATE;
    v_month_end := (DATE_TRUNC('month', CURRENT_DATE AT TIME ZONE 'America/Bogota') + INTERVAL '1 month - 1 day')::DATE;

    -- Return completeness for the month
    RETURN QUERY
    SELECT * FROM calculate_completeness(p_symbol, v_month_start, v_month_end);
END;
$$ LANGUAGE plpgsql STABLE;

-- Create indexes to optimize completeness queries
CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_date_cot
ON usdcop_m5_ohlcv (DATE(time AT TIME ZONE 'America/Bogota'), symbol);

CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_hour_dow
ON usdcop_m5_ohlcv (
    EXTRACT(HOUR FROM time AT TIME ZONE 'America/Bogota'),
    EXTRACT(DOW FROM time AT TIME ZONE 'America/Bogota')
);

-- Grant execute permissions
GRANT EXECUTE ON FUNCTION public.count_expected_trading_days(DATE, DATE) TO PUBLIC;
GRANT EXECUTE ON FUNCTION public.count_expected_bars(DATE, DATE, INTEGER) TO PUBLIC;
GRANT EXECUTE ON FUNCTION public.calculate_completeness(TEXT, DATE, DATE) TO PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_weekly_completeness(TEXT) TO PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_monthly_completeness(TEXT) TO PUBLIC;

-- Test queries (commented out for production)
-- SELECT * FROM get_weekly_completeness('USDCOP');
-- SELECT * FROM get_monthly_completeness('USDCOP');
-- SELECT * FROM calculate_completeness('USDCOP', '2025-10-01', '2025-10-23');
