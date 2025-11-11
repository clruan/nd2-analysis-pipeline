import axios from "axios";

export const apiClient = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json"
  }
});

export const api = {
  post: <T>(url: string, data?: unknown) => apiClient.post<T>(url, data).then((res) => res.data),
  get: <T>(url: string, params?: Record<string, unknown>) => apiClient.get<T>(url, { params }).then((res) => res.data)
};

