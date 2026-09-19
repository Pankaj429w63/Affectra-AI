const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}`;
    try {
      const errorData = await response.json();
      // Handle Pydantic validation errors nicely
      if (response.status === 422 && errorData.detail) {
        errorMessage = "Invalid input: " + JSON.stringify(errorData.detail);
      } else if (errorData.detail) {
        errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
      }
    } catch {
      // If we can't parse JSON, fallback to generic message
      if (response.status === 500) {
        errorMessage = 'An internal server error occurred.';
      } else if (response.status === 404) {
        errorMessage = 'The requested resource was not found.';
      }
    }
    throw new ApiError(response.status, errorMessage);
  }

  // Fast path for 204 No Content
  if (response.status === 204) {
    return {} as T;
  }

  try {
    return await response.json() as T;
  } catch {
    throw new ApiError(response.status, 'Invalid JSON response from server.');
  }
}

export const apiClient = {
  get: async <T>(endpoint: string, config?: RequestInit): Promise<T> => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...config,
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...config?.headers,
        },
      });
      return handleResponse<T>(response);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, 'Unable to connect to Affectra AI backend. Please check your connection.');
    }
  },

  post: async <T>(endpoint: string, data: unknown, config?: RequestInit): Promise<T> => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...config,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...config?.headers,
        },
        body: JSON.stringify(data),
      });
      return handleResponse<T>(response);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, 'Unable to connect to Affectra AI backend. Please check your connection.');
    }
  },

  postForm: async <T>(endpoint: string, formData: FormData, config?: RequestInit): Promise<T> => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...config,
        method: 'POST',
        headers: {
          // Do NOT set Content-Type header so browser sets multipart boundary automatically
          ...config?.headers,
        },
        body: formData,
      });
      return handleResponse<T>(response);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, 'Unable to connect to Affectra AI backend. Please check your connection.');
    }
  },
};
