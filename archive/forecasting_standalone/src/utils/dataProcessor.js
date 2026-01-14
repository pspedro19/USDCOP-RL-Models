import Papa from 'papaparse';

export const loadData = async (url) => {
    try {
        const response = await fetch(url);
        const text = await response.text();

        return new Promise((resolve, reject) => {
            Papa.parse(text, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    resolve(results.data);
                },
                error: (error) => {
                    reject(error);
                }
            });
        });
    } catch (error) {
        console.error("Error loading data:", error);
        throw error;
    }
};

export const filterData = (data, filters) => {
    if (!data) return [];

    return data.filter(row => {
        // Filter by view_type (required)
        if (row.view_type !== filters.viewType) return false;

        // Filter by inference_week (only if provided and relevant)
        if (filters.viewType === 'forward_forecast' && filters.inferenceWeek) {
            // inference_week is double in CSV usually (1.0), ensure string comparison matches
            // or match loosely
            if (row.inference_week != filters.inferenceWeek) return false;
        }

        return true;
    });
};

export const getUniqueValues = (data, field) => {
    if (!data) return [];
    const values = new Set(data.map(row => row[field]).filter(v => v != null));
    return Array.from(values).sort();
};
