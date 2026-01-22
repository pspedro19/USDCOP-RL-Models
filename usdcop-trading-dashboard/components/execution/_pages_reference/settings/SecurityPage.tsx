import { useState } from 'react'
import { Shield, Key, Lock } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { toast } from '@/stores/uiStore'

export function SecurityPage() {
  const [passwords, setPasswords] = useState({
    current: '',
    new: '',
    confirm: '',
  })
  const [isLoading, setIsLoading] = useState(false)

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()

    if (passwords.new !== passwords.confirm) {
      toast.error('Passwords do not match')
      return
    }

    if (passwords.new.length < 8) {
      toast.error('Password must be at least 8 characters')
      return
    }

    setIsLoading(true)
    // Simulate API call
    await new Promise((r) => setTimeout(r, 1000))
    setIsLoading(false)

    toast.success('Password changed successfully')
    setPasswords({ current: '', new: '', confirm: '' })
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Security"
        breadcrumbs={[
          { label: 'Settings', href: '/settings' },
          { label: 'Security' },
        ]}
      />

      {/* Change Password */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5 text-terminal-accent" />
            Change Password
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePasswordChange} className="space-y-4">
            <Input
              type="password"
              label="Current Password"
              value={passwords.current}
              onChange={(e) => setPasswords({ ...passwords, current: e.target.value })}
              leftIcon={<Lock className="h-4 w-4" />}
            />
            <Input
              type="password"
              label="New Password"
              value={passwords.new}
              onChange={(e) => setPasswords({ ...passwords, new: e.target.value })}
              leftIcon={<Lock className="h-4 w-4" />}
            />
            <Input
              type="password"
              label="Confirm New Password"
              value={passwords.confirm}
              onChange={(e) => setPasswords({ ...passwords, confirm: e.target.value })}
              leftIcon={<Lock className="h-4 w-4" />}
            />
            <Button type="submit" variant="primary" isLoading={isLoading}>
              Update Password
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Two-Factor Authentication */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-terminal-accent" />
            Two-Factor Authentication
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/50">
            <div>
              <p className="font-medium text-text-primary">Authenticator App</p>
              <p className="text-sm text-text-muted">
                Use an authenticator app for additional security
              </p>
            </div>
            <Badge variant="secondary">Coming Soon</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Sessions */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle>Active Sessions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 rounded-xl bg-slate-800/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-text-primary">Current Session</p>
                <p className="text-sm text-text-muted">
                  Browser • Last active: Just now
                </p>
              </div>
              <Badge variant="live" dot>Active</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
