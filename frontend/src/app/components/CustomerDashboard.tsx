'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';

interface CustomerData {
  [key: string]: any;
}

interface CustomerProfile {
  customerID: string;
  data: CustomerData;
}

export default function CustomerDashboard() {
  const [customerData, setCustomerData] = useState<CustomerData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [customerId, setCustomerId] = useState<string>('Ahmed'); // Default customer ID

  useEffect(() => {
    const fetchCustomerData = async () => {
      try {
        setLoading(true);
        // Fetch data from the backend API using the new customer endpoint
        const response = await fetch(`http://localhost:8098/customers/${customerId}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch customer data: ${response.statusText}`);
        }
        
        const customerProfile = await response.json();
        // Set customer data directly from the response
        setCustomerData(JSON.parse(customerProfile["data"]) || {});
        
        setError(null);
      } catch (err) {
        console.error('Error fetching customer data:', err);
        setError('Failed to load customer data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchCustomerData();
    
    // Set up polling to refresh data every 10 seconds
    const intervalId = setInterval(fetchCustomerData, 10000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, [customerId]);

  const handleCustomerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCustomerId(e.target.value);
  };

  // Helper function to render different types of data
  const renderDataItem = (key: string, value: any) => {
    if (typeof value === 'object' && value !== null) {
      return (
        <div key={key} className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h3 className="text-lg font-semibold mb-2 capitalize">{key}</h3>
          <div className="pl-4">
            {Object.entries(value).map(([subKey, subValue]) => (
              <div key={subKey} className="mb-2">
                <span className="font-medium capitalize">{subKey}:</span> {' '}
                <span>{typeof subValue === 'string' ? subValue : JSON.stringify(subValue)}</span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    
    return (
      <div key={key} className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 capitalize">{key}</h3>
        <p>{value}</p>
      </div>
    );
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Sonetto Dashboard</h1>
          <div className="flex items-center">
            <label htmlFor="customerId" className="mr-2">Customer ID:</label>
            <input
              type="text"
              id="customerId"
              value={customerId}
              onChange={handleCustomerChange}
              className="px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
            />
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-900 dark:border-white"></div>
          </div>
        ) : error ? (
          <div className="bg-red-100 dark:bg-red-900 p-4 rounded-md text-red-700 dark:text-red-100">
            {error}
          </div>
        ) : !customerData ? (
          <div className="bg-yellow-100 dark:bg-yellow-900 p-4 rounded-md text-yellow-700 dark:text-yellow-100">
            No data available for this customer.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(customerData).map(([key, value]) => renderDataItem(key, value))}
          </div>
        )}
      </div>
    </div>
  );
}