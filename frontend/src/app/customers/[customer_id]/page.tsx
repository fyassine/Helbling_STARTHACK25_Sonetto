'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { io } from 'socket.io-client';


interface Message {
  title?: string;
  content?: string;
  timestamp?: string;
  type?: string;
  [key: string]: any; // Allow for any additional properties
}

export default function CustomerPage() {
  const [message, setMessage] = useState<Message | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const params = useParams();
  const customerId = params.customer_id as string;

  useEffect(() => {
    const socket = io();
    socket.on('connect', () => {
      console.log('Connected to server');
      setConnected(true);
    });
    return () => {
      socket.disconnect();
      setConnected(false);
    };
  }, []);


  // Format timestamp if available
  const formatTime = (timestamp?: string) => {
    if (!timestamp) return '';
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch (e) {
      return timestamp;
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-indigo-50 to-blue-100 dark:from-gray-900 dark:to-indigo-950">
      <div className="w-full max-w-md">
        {/* Connection status indicator */}
        <div className="mb-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Customer: {customerId}</h1>
          <div className="flex items-center">
            <div className={`h-3 w-3 rounded-full mr-2 ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
        
        {error && (
          <div className="mb-6 p-4 bg-red-100 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-300 backdrop-blur-sm">
            {error}
          </div>
        )}
        
        {/* Glassy card UI */}
        <div className="relative overflow-hidden rounded-2xl backdrop-blur-md bg-white/30 dark:bg-gray-800/30 border border-white/50 dark:border-gray-700/50 shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.01]">
          {/* Card content */}
          <div className="p-6">
            {!message ? (
              <div className="text-center py-12">
                <p className="text-gray-500 dark:text-gray-400">Waiting for messages...</p>
              </div>
            ) : (
              <>
                {message.title && (
                  <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-white">
                    {message.title}
                  </h2>
                )}
                
                {message.content && (
                  <p className="mb-4 text-gray-700 dark:text-gray-300">
                    {message.content}
                  </p>
                )}
                
                {/* Display all other message properties */}
                <div className="mt-6 space-y-3">
                  {Object.entries(message)
                    .filter(([key]) => !['title', 'content', 'timestamp', 'type'].includes(key))
                    .map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-sm font-medium text-gray-600 dark:text-gray-400 capitalize">
                          {key.replace(/_/g, ' ')}
                        </span>
                        <span className="text-sm text-gray-800 dark:text-gray-200">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))}
                </div>
                
                {/* Message footer with type and timestamp */}
                <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700/50 flex justify-between items-center text-xs text-gray-500 dark:text-gray-400">
                  {message.type && (
                    <span className="px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                      {message.type}
                    </span>
                  )}
                  {message.timestamp && (
                    <span>{formatTime(message.timestamp)}</span>
                  )}
                </div>
              </>
            )}
          </div>
          
          {/* Decorative elements for the glassy effect */}
          <div className="absolute -top-10 -right-10 h-40 w-40 rounded-full bg-blue-300/20 dark:bg-blue-500/10 blur-xl"></div>
          <div className="absolute -bottom-8 -left-8 h-32 w-32 rounded-full bg-purple-300/20 dark:bg-purple-500/10 blur-xl"></div>
        </div>
      </div>
    </div>
  );
}