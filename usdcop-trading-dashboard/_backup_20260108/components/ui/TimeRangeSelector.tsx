/**
 * Advanced Time Range Selector Component
 * Allows navigation through years, months, days with 5-minute granularity
 */

import React, { useState, useCallback } from 'react';
import { 
  Calendar, 
  ChevronLeft, 
  ChevronRight, 
  Clock,
  ZoomIn,
  ZoomOut,
  Maximize2,
  SkipBack,
  SkipForward
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

interface TimeRangeSelectorProps {
  currentDate: Date;
  onDateChange: (date: Date) => void;
  onZoomChange: (level: 'year' | 'month' | 'week' | 'day' | '5min') => void;
  onNavigate: (direction: 'forward' | 'backward', units: number) => void;
  minDate?: Date;
  maxDate?: Date;
  isPlaying?: boolean;
}

export function TimeRangeSelector({
  currentDate,
  onDateChange,
  onZoomChange,
  onNavigate,
  minDate = new Date('2020-01-02'),
  maxDate = new Date('2025-08-22'),
  isPlaying = false
}: TimeRangeSelectorProps) {
  const [zoomLevel, setZoomLevel] = useState<'year' | 'month' | 'week' | 'day' | '5min'>('day');
  const [showDatePicker, setShowDatePicker] = useState(false);

  // Navigate by different time units
  const handleNavigate = useCallback((direction: 'forward' | 'backward', type: 'year' | 'month' | 'week' | 'day' | 'bar') => {
    let units = 1;
    
    switch(type) {
      case 'year':
        units = 252 * 59; // Trading days * bars per day (8am-12:55pm = 59 bars)
        break;
      case 'month':
        units = 21 * 59; // ~21 trading days per month
        break;
      case 'week':
        units = 5 * 59; // 5 trading days
        break;
      case 'day':
        units = 59; // 59 five-minute bars per trading day
        break;
      case 'bar':
        units = 1; // Single 5-minute bar
        break;
    }
    
    onNavigate(direction, units);
  }, [onNavigate]);

  // Quick jump to specific dates
  const quickJumps = [
    { label: 'Today', date: new Date('2025-08-22') },
    { label: '1 Week Ago', date: new Date('2025-08-15') },
    { label: '1 Month Ago', date: new Date('2025-07-22') },
    { label: '3 Months Ago', date: new Date('2025-05-22') },
    { label: '1 Year Ago', date: new Date('2024-08-22') },
    { label: 'Start', date: new Date('2020-01-02') }
  ];

  const formatDateRange = () => {
    const options: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    };
    
    return currentDate.toLocaleString('en-US', options);
  };

  return (
    <Card className="mb-4">
      <CardContent className="p-4">
        <div className="space-y-4">
          {/* Current Date Display */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Calendar className="h-5 w-5 text-gray-500" />
              <span className="text-lg font-semibold">{formatDateRange()}</span>
            </div>
            
            {/* Zoom Level Indicator */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500">View:</span>
              <select 
                value={zoomLevel}
                onChange={(e) => {
                  const level = e.target.value as typeof zoomLevel;
                  setZoomLevel(level);
                  onZoomChange(level);
                }}
                className="px-2 py-1 border rounded text-sm"
                disabled={isPlaying}
              >
                <option value="year">Year</option>
                <option value="month">Month</option>
                <option value="week">Week</option>
                <option value="day">Day</option>
                <option value="5min">5 Minutes</option>
              </select>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {/* Year Navigation */}
            <div className="flex items-center justify-center space-x-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('backward', 'year')}
                disabled={isPlaying}
                title="Previous Year"
              >
                <SkipBack className="h-3 w-3" />
                <span className="hidden md:inline">Year</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('forward', 'year')}
                disabled={isPlaying}
                title="Next Year"
              >
                <span className="hidden md:inline">Year</span>
                <SkipForward className="h-3 w-3" />
              </Button>
            </div>

            {/* Month Navigation */}
            <div className="flex items-center justify-center space-x-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('backward', 'month')}
                disabled={isPlaying}
                title="Previous Month"
              >
                <ChevronLeft className="h-3 w-3" />
                <span className="hidden md:inline">Month</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('forward', 'month')}
                disabled={isPlaying}
                title="Next Month"
              >
                <span className="hidden md:inline">Month</span>
                <ChevronRight className="h-3 w-3" />
              </Button>
            </div>

            {/* Week Navigation */}
            <div className="flex items-center justify-center space-x-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('backward', 'week')}
                disabled={isPlaying}
                title="Previous Week"
              >
                <ChevronLeft className="h-3 w-3" />
                <span className="hidden md:inline">Week</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('forward', 'week')}
                disabled={isPlaying}
                title="Next Week"
              >
                <span className="hidden md:inline">Week</span>
                <ChevronRight className="h-3 w-3" />
              </Button>
            </div>

            {/* Day Navigation */}
            <div className="flex items-center justify-center space-x-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('backward', 'day')}
                disabled={isPlaying}
                title="Previous Day"
              >
                <ChevronLeft className="h-3 w-3" />
                <span className="hidden md:inline">Day</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('forward', 'day')}
                disabled={isPlaying}
                title="Next Day"
              >
                <span className="hidden md:inline">Day</span>
                <ChevronRight className="h-3 w-3" />
              </Button>
            </div>

            {/* Bar Navigation (5-minute) */}
            <div className="flex items-center justify-center space-x-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('backward', 'bar')}
                disabled={isPlaying}
                title="Previous 5-min Bar"
              >
                <ChevronLeft className="h-3 w-3" />
                <span className="hidden md:inline">5m</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleNavigate('forward', 'bar')}
                disabled={isPlaying}
                title="Next 5-min Bar"
              >
                <span className="hidden md:inline">5m</span>
                <ChevronRight className="h-3 w-3" />
              </Button>
            </div>
          </div>

          {/* Quick Jump Buttons */}
          <div className="border-t pt-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600">Quick Jump</span>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowDatePicker(!showDatePicker)}
                disabled={isPlaying}
              >
                <Calendar className="h-3 w-3 mr-1" />
                Custom Date
              </Button>
            </div>
            
            <div className="grid grid-cols-3 md:grid-cols-6 gap-1">
              {quickJumps.map((jump) => (
                <Button
                  key={jump.label}
                  size="sm"
                  variant="ghost"
                  onClick={() => onDateChange(jump.date)}
                  disabled={isPlaying || jump.date < minDate || jump.date > maxDate}
                  className="text-xs"
                >
                  {jump.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Custom Date Picker */}
          {showDatePicker && (
            <div className="border-t pt-3">
              <input
                type="datetime-local"
                value={currentDate.toISOString().slice(0, 16)}
                onChange={(e) => {
                  const newDate = new Date(e.target.value);
                  // Validate trading hours
                  const hours = newDate.getHours();
                  const minutes = newDate.getMinutes();
                  const day = newDate.getDay();
                  
                  if (day >= 1 && day <= 5 && hours >= 8 && (hours < 13 || (hours === 12 && minutes <= 55))) {
                    onDateChange(newDate);
                    setShowDatePicker(false);
                  } else {
                    alert('Please select a time within trading hours: Monday-Friday 8:00 AM - 12:55 PM');
                  }
                }}
                min={minDate.toISOString().slice(0, 16)}
                max={maxDate.toISOString().slice(0, 16)}
                className="w-full px-3 py-2 border rounded"
              />
              <p className="text-xs text-gray-500 mt-1">
                Trading hours only: Monday-Friday 8:00 AM - 12:55 PM COT
              </p>
            </div>
          )}

          {/* Zoom Controls */}
          <div className="border-t pt-3 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onZoomChange('5min')}
                disabled={isPlaying}
                title="Zoom to 5-minute bars"
              >
                <ZoomIn className="h-3 w-3" />
                <span className="hidden md:inline ml-1">Zoom In</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onZoomChange('day')}
                disabled={isPlaying}
                title="Zoom to daily view"
              >
                <Maximize2 className="h-3 w-3" />
                <span className="hidden md:inline ml-1">Fit Day</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onZoomChange('month')}
                disabled={isPlaying}
                title="Zoom to monthly view"
              >
                <ZoomOut className="h-3 w-3" />
                <span className="hidden md:inline ml-1">Zoom Out</span>
              </Button>
            </div>
            
            <div className="text-xs text-gray-500">
              <Clock className="h-3 w-3 inline mr-1" />
              5-min bars • Mon-Fri 8:00-12:55 COT
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}